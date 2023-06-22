""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
"""

import torch
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision
from utils.metrics import estimateOverlap, estimateAccuracy
import torch.nn.functional as F
import numpy as np
from nuscenes.utils import geometry_utils

from datasets.misc_utils import get_history_frame_ids_and_masks,get_last_n_bounding_boxes
from datasets.misc_utils import update_results_bbs, generate_timestamp_prev_list


class BaseModel(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict) #调用模型，获得结果

        estimation_box = end_points['estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)
        return candidate_box

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # construct input dict
                data_dict, ref_bb = self.build_input_dict(sequence, frame_id, results_bbs)
                # run the tracker
                if torch.sum(data_dict['points'][:,:,:3]) == 0:
                    results_bbs.append(ref_bb)
                    print("Empty pointcloud!")
                else:
                    candidate_box = self.evaluate_one_sample(data_dict, ref_box=ref_bb)
                    results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)

            # # --------写出测试结果到文件---------------
            # # 获取 frame_id
            # frame_id = int(sequence[frame_id]['meta']['frame'])
            # # 获取预测结果的坐标值
            # pred_coords = results_bbs[-1].corners().T.reshape(-1)
            # # 获取真值的坐标值
            # gt_coords = this_bb.corners().T.reshape(-1)
            # # 将 frame_id、预测结果和真值合并为一个数组
            # output_data = np.hstack(([frame_id], pred_coords, gt_coords))
            # # 创建格式化字符串
            # fmt_str = "{:6.0f}, " + ", ".join(["{:15.8f}"] * 48)
            # # 将数据写入文件
            # with open('./track_result_radar_alltest_{}.txt'.format("Car"), 'a+') as f:
            #     f.write(fmt_str.format(*output_data))
            #     f.write('\n')
            # # --------写出测试结果到文件 end------------

        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, result_bbs = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success,  on_epoch=True)
        self.log('precision/test', self.prec,  on_epoch=True)
        return result_bbs

    def on_test_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

class BaseModelMF(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict) #调用模型，获得结果

        estimation_box = end_points['estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        updated_ref_boxs = end_points['ref_boxs']
        updated_ref_boxs_cpu = updated_ref_boxs.squeeze(0).detach().cpu().numpy()

        valid_mask = end_points['valid_mask'].squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)

        ref_boxs = [
            points_utils.getOffsetBB(ref_box,
                                     ref_box_cpu,
                                     degrees=self.config.degrees,
                                     use_z=self.config.use_z,
                                     limit_box=self.config.limit_box)
            for ref_box_cpu in updated_ref_boxs_cpu
        ]




        return candidate_box,ref_boxs,valid_mask

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # construct input dict
                data_dict, ref_bb = self.build_input_dict(sequence, frame_id, results_bbs)
                # run the tracker
                if torch.sum(data_dict['points'][:,:,:3]) == 0:
                    results_bbs.append(ref_bb)
                    print("Empty pointcloud!")
                else:
                    candidate_box,new_refboxs,valid_mask = self.evaluate_one_sample(data_dict, ref_box=ref_bb)
                    update_results_bbs(results_bbs,valid_mask,new_refboxs) #一定先把新ref更新进去再添加新的预测结果
                    results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)

            # # --------写出测试结果到文件---------------
            # # 获取 frame_id
            # frame_id = int(sequence[frame_id]['meta']['frame'])
            # # 获取预测结果的坐标值
            # pred_coords = results_bbs[-1].corners().T.reshape(-1)
            # # 获取真值的坐标值
            # gt_coords = this_bb.corners().T.reshape(-1)
            # # 将 frame_id、预测结果和真值合并为一个数组
            # output_data = np.hstack(([frame_id], pred_coords, gt_coords))
            # # 创建格式化字符串
            # fmt_str = "{:6.0f}, " + ", ".join(["{:15.8f}"] * 48)
            # # 将数据写入文件
            # with open('./track_result_radar_alltest_{}.txt'.format("Car"), 'a+') as f:
            #     f.write(fmt_str.format(*output_data))
            #     f.write('\n')
            # # --------写出测试结果到文件 end------------

        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, result_bbs = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success,  on_epoch=True)
        self.log('precision/test', self.prec,  on_epoch=True)
        return result_bbs

    def on_test_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.current_epoch) #上报大指标直接用epoch

class BaseModelImage(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict) #调用模型，获得结果

        estimation_box = end_points['estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)
        return candidate_box

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # construct input dict
                data_dict, ref_bb = self.build_input_dict(sequence, frame_id, results_bbs)
                # run the tracker
                candidate_box = self.evaluate_one_sample(data_dict, ref_box=ref_bb)
                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)

            # # --------写出测试结果到文件---------------
            # # 获取 frame_id
            # frame_id = int(sequence[frame_id]['meta']['frame'])
            # # 获取预测结果的坐标值
            # pred_coords = results_bbs[-1].corners().T.reshape(-1)
            # # 获取真值的坐标值
            # gt_coords = this_bb.corners().T.reshape(-1)
            # # 将 frame_id、预测结果和真值合并为一个数组
            # output_data = np.hstack(([frame_id], pred_coords, gt_coords))
            # # 创建格式化字符串
            # fmt_str = "{:6.0f}, " + ", ".join(["{:15.8f}"] * 48)
            # # 将数据写入文件
            # with open('./track_result_radar_alltest_{}.txt'.format("Car"), 'a+') as f:
            #     f.write(fmt_str.format(*output_data))
            #     f.write('\n')
            # # --------写出测试结果到文件 end------------

        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, result_bbs = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success,  on_epoch=True)
        self.log('precision/test', self.prec,  on_epoch=True)
        return result_bbs

    def on_test_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

class MatchingBaseModel(BaseModel):

    def compute_loss(self, data, output):
        """

        :param data: input data
        :param output:
        :return:
        """
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        estimation_cla = output['estimation_cla']  # B,N
        seg_label = data['seg_label']
        box_label = data['box_label']  # B,4
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]

        loss_seg = F.binary_cross_entropy_with_logits(estimation_cla, seg_label)

        loss_vote = F.smooth_l1_loss(vote_xyz, box_label[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
        loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        objectness_label[dist < 0.3] = 1
        objectness_score = estimation_boxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                            pos_weight=torch.tensor([2.0]).cuda())
        loss_objective = torch.sum(loss_objective * objectness_mask) / (
                torch.sum(objectness_mask) + 1e-6)
        loss_box = F.smooth_l1_loss(estimation_boxes[:, :, :4],
                                    box_label[:, None, :4].expand_as(estimation_boxes[:, :, :4]),
                                    reduction='none')
        loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
        }

    def generate_template(self, sequence, current_frame_id, results_bbs):
        """
        generate template for evaluating.
        the template can be updated using the previous predictions.
        :param sequence: the list of the whole sequence
        :param current_frame_id:
        :param results_bbs: predicted box for previous frames
        :return:
        """
        first_pc = sequence[0]['pc']
        previous_pc = sequence[current_frame_id - 1]['pc']
        if "firstandprevious".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([first_pc, previous_pc],
                                                               [results_bbs[0], results_bbs[current_frame_id - 1]],
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        elif "first".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(first_pc, results_bbs[0],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "previous".upper() in self.config.hape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(previous_pc, results_bbs[current_frame_id - 1],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "all".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([frame["pc"] for frame in sequence[:current_frame_id]],
                                                               results_bbs,
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        return template_pc, canonical_box

    def generate_search_area(self, sequence, current_frame_id, results_bbs):
        """
        generate search area for evaluating.

        :param sequence:
        :param current_frame_id:
        :param results_bbs:
        :return:
        """
        this_bb = sequence[current_frame_id]["3d_bbox"]
        this_pc = sequence[current_frame_id]["pc"]
        if ("previous_result".upper() in self.config.reference_BB.upper()):
            ref_bb = results_bbs[-1]
        elif ("previous_gt".upper() in self.config.reference_BB.upper()):
            previous_bb = sequence[current_frame_id - 1]["3d_bbox"]
            ref_bb = previous_bb
        elif ("current_gt".upper() in self.config.reference_BB.upper()):
            ref_bb = this_bb
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset)
        return search_pc_crop, ref_bb

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
        """
        construct input dict for evaluating
        :param template_pc:
        :param search_pc:
        :param template_box:
        :return:
        """
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
        }
        return data_dict

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        # preparing search area
        search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)
        # update template
        template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)
        # construct input dict
        data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)
        return data_dict, ref_bb


class MotionBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs): #注意：可能会有空点云输入
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]
        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        ref_box = results_bbs[-1]
        prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_box,
                                                        scale=self.config.bb_scale, #多次搜索区域为空是不是要扩大一下搜索区域？
                                                        offset=self.config.bb_offset)
        this_frame_pc = points_utils.generate_subwindow(this_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        canonical_box = points_utils.transform_box(ref_box, ref_box)
        prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征

        this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征
        seg_mask_prev = geometry_utils.points_in_box(canonical_box, prev_points.T[:3,:], 1.25).astype(float)

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

        timestamp_prev = np.full((self.config.point_sample_size, 1), fill_value=0)
        timestamp_this = np.full((self.config.point_sample_size, 1), fill_value=0.1)
        prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points = np.concatenate([prev_points, this_points], axis=0)

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32),
                     }
        if getattr(self.config, 'box_aware', False):
            candidate_bc_prev = points_utils.get_point_to_box_distance(
                stack_points[:self.config.point_sample_size, :3], canonical_box)
            candidate_bc_this = np.zeros_like(candidate_bc_prev)
            candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)
            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]

class MotionBaseModelMF(BaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs): #注意：可能会有空点云输入
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        #整体思路：
        # 获取多个历史点云
        # 获取多个历史框

        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(frame_id,self.hist_num)
        prev_frames = [sequence[id] for id in prev_frame_ids]
        this_frame = sequence[frame_id]
        this_pc = this_frame['pc']
        bbox_size = this_frame['3d_bbox'].wlh
        prev_pcs = [frame['pc'] for frame in prev_frames]
        ref_boxs = get_last_n_bounding_boxes(results_bbs,valid_mask)
        num_hist = len(valid_mask)

        prev_frame_pcs = []
        for i, prev_pc in enumerate(prev_pcs):
            prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_boxs[0],
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)
            prev_frame_pcs.append(prev_frame_pc)

        this_frame_pc = points_utils.generate_subwindow(this_pc, ref_boxs[0],
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        # canonical_box = points_utils.transform_box(ref_boxs[0], ref_boxs[0])
        ref_boxs = [
            points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs
        ]

        prev_points_list = [points_utils.regularize_pc(prev_frame_pc.points.T, self.config.point_sample_size)[0] for prev_frame_pc in prev_frame_pcs] #采样到特定数量,这里的策略是在已有的点里面重复随机选，直到达到特定数量

        this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征
        seg_mask_prev_list = [geometry_utils.points_in_box(ref_box, prev_points.T[:3,:], 1.25).astype(float) for ref_box,prev_points in zip(ref_boxs,prev_points_list)]#应当只考虑xyz特征

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            for seg_mask_prev in seg_mask_prev_list:
                # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
                # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
                seg_mask_prev[seg_mask_prev == 0] = 0.2
                seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)

        timestamp_prev_list = generate_timestamp_prev_list(valid_mask,self.config.point_sample_size)
        timestamp_this = np.full((self.config.point_sample_size, 1), fill_value=0.1)
        prev_points_list = [
        np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]],
                       axis=-1)
        for prev_points, timestamp_prev, seg_mask_prev in zip(
            prev_points_list, timestamp_prev_list, seg_mask_prev_list)
        ]

        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points_list = prev_points_list + [this_points]
        stack_points = np.concatenate(stack_points_list, axis=0)

        #生成refbox
        ref_box_thetas = [
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if self.config.degrees else ref_box.orientation.radians *
            ref_box.orientation.axis[-1] for ref_box in ref_boxs
        ]
        ref_box_list = [
            np.append(ref_box.center,
                      theta).astype('float32') for ref_box, theta in zip(
                          ref_boxs, ref_box_thetas)
        ]
        ref_boxs_np = np.stack(ref_box_list, axis=0)

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32), #都要调整维度，生成B这个维度
                     "ref_boxs":torch.tensor(ref_boxs_np[None, :], device=self.device, dtype=torch.float32), #都要调整维度，生成B这个维度
                     "valid_mask":torch.tensor(valid_mask, device=self.device, dtype=torch.float32).unsqueeze(0), #都要调整维度，生成B这个维度
                     "bbox_size":torch.tensor(bbox_size[None, :],device=self.device, dtype=torch.float32),
                     }

        if getattr(self.config, 'box_aware', False):
            stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
            hist_points_list = stack_points_split[:num_hist] # 仅保留前 3 个历史帧的点云
            candidate_bc_prev_list= [
                points_utils.get_point_to_box_distance(hist_points[:, :3], ref_box)
                for hist_points, ref_box in zip(hist_points_list, ref_boxs)
            ]
            candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
            candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
            candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]

class MotionBaseModelRadarLidar(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs): #注意：可能会有空点云输入
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]
        prev_radar_pc, prev_lidar_pc = prev_frame['radar_pc'], prev_frame['lidar_pc']
        this_radar_pc, this_lidar_pc = this_frame['radar_pc'], this_frame['lidar_pc']
        ref_box = results_bbs[-1]
        this_box = this_frame['3d_bbox']
        prev_frame_radar_pc = points_utils.generate_subwindow(prev_radar_pc, ref_box,
                                                        scale=self.config.bb_scale, #多次搜索区域为空是不是要扩大一下搜索区域？
                                                        offset=self.config.bb_offset)
        this_frame_radar_pc = points_utils.generate_subwindow(this_radar_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        prev_frame_lidar_pc = points_utils.generate_subwindow(prev_lidar_pc, ref_box,
                                                        scale=self.config.bb_scale, #多次搜索区域为空是不是要扩大一下搜索区域？
                                                        offset=self.config.bb_offset)
        this_frame_lidar_pc = points_utils.generate_subwindow(this_lidar_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        canonical_box = points_utils.transform_box(ref_box, ref_box)
        local_box = points_utils.transform_box(this_box, ref_box)
        prev_radar_points, idx_prev = points_utils.regularize_pc(prev_frame_radar_pc.points.T,
                                                           self.config.radar_point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征

        this_radar_points, idx_this = points_utils.regularize_pc(this_frame_radar_pc.points.T,
                                                           self.config.radar_point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征

        prev_lidar_points, idx_prev = points_utils.regularize_pc(prev_frame_lidar_pc.points.T,
                                                           self.config.lidar_point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征
        this_lidar_points, idx_this = points_utils.regularize_pc(this_frame_lidar_pc.points.T,
                                                           self.config.lidar_point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征



        seg_mask_prev = geometry_utils.points_in_box(canonical_box, prev_lidar_points.T[:3,:], 1.25).astype(float)
        seg_label_prev = geometry_utils.points_in_box(canonical_box, prev_lidar_points.T[:3,:], 1.25).astype(int)
        seg_label_this = geometry_utils.points_in_box(local_box, this_lidar_points.T[:3,:], 1.25).astype(int)
        stack_seg_label = np.hstack([seg_label_prev, seg_label_this])

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

        timestamp_prev = np.full((self.config.lidar_point_sample_size, 1), fill_value=0)
        timestamp_this = np.full((self.config.lidar_point_sample_size, 1), fill_value=0.1)
        prev_points = np.concatenate([prev_lidar_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
        this_points = np.concatenate([this_lidar_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points = np.concatenate([prev_points, this_points], axis=0)

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32),
                     "seg_label": torch.tensor(stack_seg_label[None, :], device=self.device, dtype=torch.int),
                     'radar_points_prev': torch.tensor(prev_radar_points[None, :], device=self.device, dtype=torch.float32),
                     'radar_points_this': torch.tensor(this_radar_points[None, :], device=self.device, dtype=torch.float32), }
        if getattr(self.config, 'box_aware', False):
            candidate_bc_prev = points_utils.get_point_to_box_distance(
                stack_points[:self.config.lidar_point_sample_size, :3], canonical_box)
            candidate_bc_this = np.zeros_like(candidate_bc_prev)
            candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)
            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]

class MotionBaseModelImage(BaseModelImage):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs): #注意：可能会有空点云输入
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]
        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        prev_masked_image = prev_frame['masked_image']
        this_masked_image = prev_frame['masked_image']
        ref_box = results_bbs[-1]
        prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_box,
                                                        scale=self.config.bb_scale, #多次搜索区域为空是不是要扩大一下搜索区域？
                                                        offset=self.config.bb_offset)
        this_frame_pc = points_utils.generate_subwindow(this_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        canonical_box = points_utils.transform_box(ref_box, ref_box)
        prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征

        this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1) #获得统一数量的点，如果点为空，则返回全0特征
        seg_mask_prev = geometry_utils.points_in_box(canonical_box, prev_points.T[:3,:], 1.25).astype(float)

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

        timestamp_prev = np.full((self.config.point_sample_size, 1), fill_value=0)
        timestamp_this = np.full((self.config.point_sample_size, 1), fill_value=0.1)
        prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points = np.concatenate([prev_points, this_points], axis=0)

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32),
                     "prev_masked_image":torch.tensor(prev_masked_image[None,:], device=self.device, dtype=torch.float32),
                     "this_masked_image":torch.tensor(this_masked_image[None,:], device=self.device, dtype=torch.float32),
                     }
        if getattr(self.config, 'box_aware', False):
            candidate_bc_prev = points_utils.get_point_to_box_distance(
                stack_points[:self.config.point_sample_size, :3], canonical_box)
            candidate_bc_this = np.zeros_like(candidate_bc_prev)
            candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)
            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]