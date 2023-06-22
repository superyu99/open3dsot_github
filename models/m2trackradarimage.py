"""
m2track.py
Created by zenn at 2021/11/24 13:10
"""
from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet

import torch
from torch import nn
import torch.nn.functional as F

from utils.metrics import estimateOverlap, estimateAccuracy
from torchmetrics import Accuracy

from models.backbone.image_backbone import build_backbone
from models.image_transformer import ImageAwareTransformer

class M2TRACKRADARIMAGE(base_model.MotionBaseModelImage):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.seg_acc = Accuracy(task='multiclass',num_classes=2, average='none')

        # --------------------------------point --------------------------------------------------------
        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_second_stage = getattr(config, 'use_second_stage', True)
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', True)
        #3 + 3 + 1 + 1 解释：
        # 3：xyz
        # 3：其他特征
        # 1：时间戳
        # 1：概率
        self.seg_pointnet = SegPointNet(input_channel=3 + 3 + 1 + 1 + (9 if self.box_aware else 0), #此处注意输入维度
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        #3 + 3 + 1 解释：
        # 3：xyz
        # 3：其他特征
        # 1：时间戳
        self.mini_pointnet = MiniPointNet(input_channel=3 + 3 + 1 + (9 if self.box_aware else 0), ##此处注意输入维度
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
        # --------------------------------point end--------------------------------------------------------

        # -------------------------------- image --------------------------------------------------------
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_queries = getattr(config, 'num_queries', 1) #1 仅需预测一个偏移
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 2) #！！！1,512
        

        self.image_backbone = build_backbone(config)
        self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.image_backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])

        self.reference_points = nn.Linear(self.hidden_dim, 2)
        self.image_aware_transformer = ImageAwareTransformer()


        # -------------------------------- image end --------------------------------------------------------

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)

        }

        Returns: B,4

        """
        #-------------------------image-----------------------------------
        image_prev = input_dict["prev_masked_image"].permute(0,3,1,2) #CHW
        image_this = input_dict["this_masked_image"].permute(0,3,1,2) #CHW
        prev_img_feature,prev_img_pos = self.image_backbone(image_prev) #torch.Size([2, 1024, 76, 121])
        this_img_feature,this_img_pos = self.image_backbone(image_this) #torch.Size([2, 1024, 76, 121])

        feat_prev,mask_prev = prev_img_feature[0].decompose()
        feat_this,mask_this = this_img_feature[0].decompose()

        feat = feat_this - feat_prev
        pos = this_img_pos[0] - prev_img_pos[0]


        src = self.input_proj[0](feat)

        #-------------------------image end-------------------------------
        #-------------------------point-----------------------------------
        output_dict = {}
        x = input_dict["points"].transpose(1, 2) #torch.Size([1, 1024, 8])
        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2) #box角点到每个点的距离特征
            x = torch.cat([x, candidate_bc], dim=1)

        B, _, N = x.shape

        seg_out = self.seg_pointnet(x) #torch.Size([1, 11, 1024])
        seg_logits = seg_out[:, :2, :]  # B,2,N #选出概率
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        # 7 的解释，3+3+1 3：xyz， 3：其他特征 1：时间戳
        # 4 的解释，3+1 3：xyz，  1：时间戳
        mask_points = x[:, :7, :] * pred_cls #取出原始特征，赋上权重，此处注意维度，输入多了3个维度
        mask_xyz_t0 = mask_points[:, :3, :N // 2]  # B,3,N//2
        mask_xyz_t1 = mask_points[:, :3, N // 2:]
        if self.box_aware:
            pred_bc = seg_out[:, 2:, :] #所以说这个输出可以被认为是9个（8corner+1center）点的特征
            mask_pred_bc = pred_bc * pred_cls
            # mask_pred_bc_t0 = mask_pred_bc[:, :, :N // 2]  # B,9,N//2
            # mask_pred_bc_t1 = mask_pred_bc[:, :, N // 2:]
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        point_feature = self.mini_pointnet(mask_points) #用于提取点特征 [B, 256]

        #--------------------------transformer融合-----------------------------
        query_embed = self.query_embed.weight #1*512
        #     tgt,  #query
        #     img_feature,  #图像特征,也就是SRC
        #     image_pos,  #图像pos
        #     tgt_pos_embed,  #与query同源的pos
        #     points_embed,  #points backbone的特征
        query_embed, tgt = torch.split(query_embed, self.hidden_dim, dim=1)
        tgt = tgt.unsqueeze(0).expand(B, -1, -1) #torch.Size([2, 50, 256]),为每一个batch都复制一份
        reference_points = self.reference_points(query_embed).sigmoid() #torch.Size([2, 50, 2]) 参考点竟然是由query生成的

        query = self.image_aware_transformer(
            tgt,
            src,
            pos,
            query_embed,
            point_feature,
            reference_points,
        )
        #--------------------------transformer融合-----------------------------

        # motion state prediction
        # motion_pred = self.motion_mlp(point_feature)  # B,4 这里的输出是 delta box，即为两个box之间的motion
        motion_pred = self.motion_mlp(query.squeeze(1))  # B,4 这里的输出是 delta box，即为两个box之间的motion
        if self.use_motion_cls: #用于监督
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits
        else:
            motion_pred_masked = motion_pred
        # previous bbox refinement
        if self.use_prev_refinement: #用于监督
            prev_boxes = self.final_mlp(point_feature)  # 同时预测出前一帧的box B,4 
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred) #无论是在训练阶段还是测试阶段，他都不需要前一帧的box，前一帧的box需要用的时候是预测出来的，要么就是0

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked) #motion_pred预测的是相对的，需要放在坐标系里才能出来一个box

        # 2nd stage refinement
        if self.use_second_stage:
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2), #以前帧的box为基准，依据预测得到的一阶段motion，把前一帧的点平移
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)  # B,3,N//2
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,N #拼起来

            # transform to the aux_box coordinate system 把坐标系也平移到一阶段出来的deltabox中心
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
                                                                       aux_box).transpose(1, 2)

            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)
            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4 再来一次，出一个box
            output = points_utils.get_offset_box_tensor(aux_box, output_offset) #auxbox所处的坐标系已经是以上一帧box为中心了，是可以直接作为输出值的
            output_dict["estimation_boxes"] = output
        else:
            output_dict["estimation_boxes"] = aux_box
        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': aux_box,
                            })

        return output_dict

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits']
        with torch.no_grad():
            seg_label = data['seg_label']
            box_label = data['box_label']
            box_label_prev = data['box_label_prev']
            motion_label = data['motion_label']
            motion_state_label = data['motion_state_label']
            center_label = box_label[:, :3]
            angle_label = torch.sin(box_label[:, 3])
            center_label_prev = box_label_prev[:, :3]
            angle_label_prev = torch.sin(box_label_prev[:, 3])
            center_label_motion = motion_label[:, :3]
            angle_label_motion = torch.sin(motion_label[:, 3])

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6)
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

        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (
                              loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight)
        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
        })
        if self.box_aware:
            prev_bc = data['prev_bc']
            this_bc = data['this_bc']
            bc_label = torch.cat([prev_bc, this_bc], dim=1)
            pred_bc = output['pred_bc']
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
                                         batch['motion_state_label'])
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        return loss


