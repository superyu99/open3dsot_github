"""
main.py
Created by zenn at 2021/7/18 15:08
"""
import pytorch_lightning as pl
import argparse

# import pytorch_lightning.utilities.distributed
import torch
import yaml
from easydict import EasyDict
import os

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import get_model


from datasets.data_classes import Box
from pyquaternion import Quaternion
import numpy as np

from utils import platform_utils
if platform_utils.USE_COMPUTER == True:
    import vis_tool as vt

torch.set_float32_matmul_precision("high")

# os.environ["NCCL_DEBUG"] = "INFO"

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--save_top_k', type=int, default=-1, help='save top k checkpoints')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--cfg', type=str, help='the config_file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)


cfg = parse_config()
env_cp = os.environ.copy()
try:
    node_rank, local_rank, world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']

    is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
    pl_trainer_gpus = env_cp['PL_TRAINER_GPUS']
    print(node_rank, local_rank, world_size, is_in_ddp_subprocess, pl_trainer_gpus)

    if int(local_rank) == int(world_size) - 1:
        print(cfg)
except KeyError:
    pass

# init model
if cfg.checkpoint is None:
    net = get_model(cfg.net_model)(cfg)
else:
    net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
if not cfg.test:
    # dataset and dataloader
    train_data = get_dataset(cfg, type=cfg.train_type, split=cfg.train_split)
    # tracks =  train_data.dataset.tracklet_anno_list
    # for track in tracks:
    #     for sample in track:
    #         info = train_data.dataset._get_frame_from_anno_data(sample)
            # points = info["pc"].points.T - info["3d_bbox"].center
            # box = info["3d_bbox"].corners().T - info["3d_bbox"].center
            # vt.show_scenes(hist_pointcloud=[points],bboxes=[box])#,gt_bbox=[prev_box.corners().T])

    # for i in range(5350,len(train_data),1):
    #     data = train_data.__getitem__(i)
        # center = data["box_label"][:3]
        # size = data["bbox_size"]
        # rotation = data["box_label"][3]
        # # print(rotation)
        # orientation = Quaternion(
        #         axis=[0, 0, 1], radians=rotation) 
        # box = Box(center,size,orientation)
                

    #     prev_center = data["box_label_prev"][:3]
    #     prev_size = data["bbox_size"]
    #     prev_rotation = data["box_label_prev"][3]
    #     print(prev_rotation)
    #     prev_orientation = Quaternion(
    #             axis=[0, 0, 1], radians=prev_rotation) 
    #     prev_box = Box(prev_center,prev_size,prev_orientation)
        # if platform_utils.USE_COMPUTER == True and data["points"].shape[0]>100:
        #     vt.show_scenes(hist_pointcloud=[data["lidar_points_this"]],bboxes=[box.corners().T])#,gt_bbox=[prev_box.corners().T])
    val_data = get_dataset(cfg, type='test', split=cfg.val_split)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=True,drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    checkpoint_callback = ModelCheckpoint(monitor='precision/test', mode='max', save_last=True,
                                          save_top_k=cfg.save_top_k)

    # init trainer
    trainer = pl.Trainer(devices=-1, accelerator='auto', max_epochs=cfg.epoch,
                         callbacks=[checkpoint_callback],
                         default_root_dir=cfg.log_dir,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                         num_sanity_val_steps=0,
                         gradient_clip_val=cfg.gradient_clip_val,
                         fast_dev_run=False)
    trainer.fit(net, train_loader, val_loader, ckpt_path=cfg.checkpoint)
else:
    test_data = get_dataset(cfg, type='test', split=cfg.test_split)
    # for i in range(len(test_data)):
    #     data = test_data.__getitem__(i)
    # for i in range(len(test_data)):
    #     data = test_data.__getitem__(i)
    #     for j in range(len(data)):
    #         box = data[j]["3d_bbox"]
    #         center = box.center
    #         size = box.wlh
    #         rotation = box.rotation_matrix[0][1]
    #         print(rotation)
    #         orientation = Quaternion(
    #                 axis=[0, 0, -1], radians=rotation) 
    #         box = Box(center,size,orientation)
                    

    #         # prev_center = data["box_label_prev"][:3]
    #         # prev_size = data["bbox_size"]
    #         # prev_rotation = data["box_label_prev"][3]
    #         # print(prev_rotation)
    #         # prev_orientation = Quaternion(
    #         #         axis=[0, 0, 1], radians=prev_rotation) 
    #         # prev_box = Box(prev_center,prev_size,prev_orientation)
    #         print(data[j]["pc"].points.T.shape)

    #         vt.show_scenes(hist_pointcloud=[data[j]["pc"].points.T],bboxes=[box.corners().T])
    test_loader = DataLoader(test_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)

    trainer = pl.Trainer(devices=-1, accelerator='auto', default_root_dir=cfg.log_dir)
    trainer.test(net, test_loader, ckpt_path=cfg.checkpoint)
