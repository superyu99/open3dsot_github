"""
m2track.py
Created by zenn at 2021/11/24 13:10
"""
from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet
from models.attn.Models import Transformer

import torch
from torch import nn
import torch.nn.functional as F

from utils.metrics import estimateOverlap, estimateAccuracy
from torchmetrics import Accuracy

from datasets.misc_utils import get_tensor_corners
from datasets.misc_utils import create_corner_timestamps


class M2TRACKMFATTN(base_model.MotionBaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.hist_num = getattr(config, 'hist_num', 1)
        self.seg_acc = Accuracy(task='multiclass',num_classes=2, average='none')

        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_second_stage = getattr(config, 'use_second_stage', False)
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', False)
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)
        if self.use_second_stage:
            self.mini_pointnet2 = MiniPointNet(input_channel=3 + (9 if self.box_aware else 0),
                                               per_point_mlp=[64, 128, 256, 512],
                                               hidden_mlp=[512, 256],
                                               output_size=-1)

            self.box_mlp = nn.Sequential(nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 4))
        if self.use_prev_refinement:
            self.final_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))
        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = Accuracy(task='multiclass',num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))
        #---------------------transformer--------------------------------------
        self.Transformer = Transformer(d_word_vec=64, d_model=64, d_inner=512,
            n_layers=1, n_head=4, d_k=64, d_v=64, n_position = 1024*4)

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)
            ['points', #[2, 4096, 5] B*(num_hist*sample)*5
            'box_label', #B*4
            'ref_boxs', #B*(num_hist)*4
            'box_label_prev', #B*(num_hist)*4
            'motion_label', #B*(num_hist)*4
            'motion_state_label', #B*(num_hist) 当前的box减去之前所有的histbox
            'bbox_size', #B*3
            'seg_label', #B*(num_hist+1)*sample
            'valid_mask', #B*(num_hist)
            'prev_bc', #B*(num_hist)*sample*9
            'this_bc', #B*sample*9
            'candidate_bc'] #B*(num_hist*sample)*9

        }

        Returns: B,4

        """
        output_dict = {}
        x = input_dict["points"].transpose(1, 2) # torch.Size([1, 5, 4096]) #4096前3部分是历史 最后是当前
        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2) #torch.Size([1, 9, 4096])
            x = torch.cat([x, candidate_bc], dim=1) #torch.Size([1, 14, 4096]) 3+2+9

        B, _, N = x.shape
        HL =  input_dict["valid_mask"].shape[1] #历史帧个数
        L = HL + 1 #总的点云序列的长度，1代表当前帧
        chunk_size = N // L

        ##此处可能会成为瓶颈！！---------------------
        seg_out = self.seg_pointnet(x) #torch.Size([1, 11, 4096]) 
        seg_logits = seg_out[:, :2, :]  # B,2,N
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls #只拿了前4维数据，并且直接乘上概率 前4维数据的含义：x,y,z,time_stamp

        mask_points_chunks = torch.chunk(mask_points, L, dim=-1) #划分成L份
        mask_xyz_t0 = mask_points_chunks[0][:,:3]  # 获取第一个历史帧的mask # B,3,1024
        mask_xyz_t1 = mask_points_chunks[-1][:,:3] # 获取当前帧的mask # B,3,1024

        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            # mask_pred_bc_t0 = mask_pred_bc[:, :, :N // 2]  # B,9,N//2
            # mask_pred_bc_t1 = mask_pred_bc[:, :, N // 2:]
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        #基于历史的第一帧和当前帧作motion预测 
        mask_points_new = torch.cat((mask_points[:, :, :chunk_size], mask_points[:, :, -chunk_size:]), dim=-1)
        point_feature = self.mini_pointnet(mask_points_new) #N*256

        # motion state prediction
        motion_pred = self.motion_mlp(point_feature)  # B,4
        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits # B*2
        else:
            motion_pred_masked = motion_pred
        # previous bbox refinement
        if self.use_prev_refinement:
            prev_boxes = self.final_mlp(point_feature)  # previous bb, B,4
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        #------------对历史的box和当前预测的box作纠正-----------------
        #获取当前box和历史box的corners
        bbox_size = input_dict["bbox_size"][0] #同一个batch的形状是一样的
        ref_boxs = input_dict["ref_boxs"]
        box_seq = torch.cat((ref_boxs, aux_box.unsqueeze(1)), dim=1) #与生成数据集的时候顺序一致拼接，先历史后当前，历史第一帧是当前帧前面的那一帧
        box_seq = box_seq.reshape(-1,4) #把box个数合并到B，因为都是box，统一对待即可
        box_seq_corner_list = [get_tensor_corners(box[:3],bbox_size,box[-1]).T for box in box_seq]
        box_seq_corners = torch.cat(box_seq_corner_list, dim=0).reshape(B,L*8,-1) #B*(L*8)*3 一共L*8个点，每个点3个特征
        #给boxcorners加上数间戳特征
        corner_stamps = create_corner_timestamps(B,HL,8).to(self.device)
        box_seq_corners = torch.cat((box_seq_corners,corner_stamps),dim=-1) #B*(L*8)*4 4代表xyz和时间戳特征

        delta_motion = self.Transformer(box_seq_corners,input_dict["points"],input_dict["valid_mask"])  #B*4*4

        #为历史的refbox加上修正量，去做loss
        updated_ref_boxs = ref_boxs + delta_motion[:,:HL,:]
        #为当前的预测加上修正量，作为最终输出
        aux_box = aux_box + delta_motion[:,-1,:]

        #---------------------------------------------------------

        # 2nd stage refinement
        if self.use_second_stage:
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2),
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)  # B,3,N//2
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,N

            # transform to the aux_box coordinate system
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
                                                                       aux_box).transpose(1, 2)

            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)
            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4
            output = points_utils.get_offset_box_tensor(aux_box, output_offset)
            # output_dict["estimation_boxes"] = output
            output_dict["estimation_boxes"] = input_dict["box_label"]
        else:
            output_dict["estimation_boxes"] = aux_box
            # output_dict["estimation_boxes"] = input_dict["box_label"]
        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': aux_box,
                            'ref_boxs': input_dict['ref_boxs'],
                            'valid_mask':input_dict["valid_mask"],
                            'updated_ref_boxs':updated_ref_boxs,
                            })

        return output_dict

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits'] #[B, 2, 4096]
        updated_ref_boxs = output['updated_ref_boxs']
        with torch.no_grad():
            seg_label = data['seg_label'] #[2, 4096]
            box_label = data['box_label'] #[2, 4]
            box_label_prev = data['box_label_prev'] #torch.Size([2, 3, 4])
            motion_label = data['motion_label'] #torch.Size([2, 3, 4])
            motion_state_label = data['motion_state_label'][:,0] # 0代表目前仅仅约束与第一个历史box的motion torch.Size([2, 3])------------
            center_label = box_label[:, :3] #torch.Size([2, 3])
            angle_label = torch.sin(box_label[:, 3]) #torch.Size([2])
            center_label_prev = box_label_prev[:, :3] #torch.Size([2, 3, 4])
            angle_label_prev = torch.sin(box_label_prev[:,0,3]) #0代表第一个历史box的角度，暂时这样------------------
            center_label_motion = motion_label[:,0,:3] #0代表与第一个历史box的motion，暂时这样------------------
            angle_label_motion = torch.sin(motion_label[:,0,3]) #0代表与第一个历史box的motion，暂时这样------------------

            #----参考boxlabel 计算refbox的loss
            ref_label = data['box_label_prev']
            ref_center_label = ref_label[:, :, :3] #B*hist_num*3
            ref_angle_label = torch.sin(ref_label[:,:,3]) #B*hist_num 3角度的sin值


        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6) #在一个batch内作平衡
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)

        if self.use_second_stage:
            estimation_boxes = output['estimation_boxes']  # B,4
            loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
            loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
            loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
            loss_dict["loss_center"] = loss_center
            loss_dict["loss_angle"] = loss_angle
        if self.use_prev_refinement:
            estimation_boxes_prev = output['estimation_boxes_prev']  # B,4
            loss_center_prev = F.smooth_l1_loss(estimation_boxes_prev[:, :3], center_label_prev)
            loss_angle_prev = F.smooth_l1_loss(torch.sin(estimation_boxes_prev[:, 3]), angle_label_prev)
            loss_total += (loss_center_prev * self.config.center_weight + loss_angle_prev * self.config.angle_weight)
            loss_dict["loss_center_prev"] = loss_center_prev
            loss_dict["loss_angle_prev"] = loss_angle_prev

        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)


        #---------------------refbox loss--------------------------
        loss_center_ref = F.smooth_l1_loss(updated_ref_boxs[:,:,:3],ref_center_label)
        loss_angle_ref = F.smooth_l1_loss(torch.sin(updated_ref_boxs[:, :, 3]), ref_angle_label)
        #---------------------refbox loss--------------------------


        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight) \
                      + 1 * (loss_center_ref * self.config.ref_center_weight + loss_angle_ref * self.config.ref_angle_weight)
                    
        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
            "loss_center_ref": loss_center_ref,
            "loss_angle_ref": loss_angle_ref,
        })
        if self.box_aware:
            # prev_bc = data['prev_bc'] #torch.Size([2, 3, 1024, 9])
            prev_bc = torch.flatten(data['prev_bc'], start_dim=1, end_dim=2) #我直接调整了，暂时这样-------------------
            this_bc = data['this_bc'] #torch.Size([2, 1024, 9])
            bc_label = torch.cat([prev_bc, this_bc], dim=1) #torch.Size([2, 4096, 9])
            pred_bc = output['pred_bc'] #torch.Size([2, 4096, 9])
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        # log
        seg_acc = self.seg_acc(torch.argmax(output['seg_logits'], dim=1, keepdim=False), batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(output['motion_cls'], dim=1, keepdim=False),
                                         batch['motion_state_label'][:,0]) #0代表与第一个历史box的motion--------------------
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)

        return loss
