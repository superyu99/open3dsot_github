# Created by zenn at 2021/4/27

import copy
import random

from torch.utils.data import Dataset
from datasets.data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import warnings
import pickle
from collections import defaultdict
from datasets import points_utils, base_dataset

from vod.configuration import KittiLocations
import torch
import scipy
from scipy.spatial import Delaunay, ConvexHull
from vod.frame import FrameDataLoader, FrameTransformMatrix, FrameLabels, transform_pcl_with_all_feature
#----------------------utils---------------------------
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p[:,0:3]) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

# 给出坐标系之间的变换矩阵，将xyzwhlry格式的box转化到另一个坐标系的函数
# ly-2023-0510
def transform_3dbox(box, trans_matrix):
    x, y, z, l, h, w, rotation = box
    
    # 创建一个表示3D框中心点的向量
    center = np.array([x, y, z, 1])
    
    # 将中心点从一个坐标系转换到另一个坐标系
    transformed_center = (trans_matrix @ center)[:3]
    
    # 取消对旋转所做的更改
    rotation = -(rotation + np.pi / 2) #这一步与原版对应
    
    # 计算旋转矩阵
    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                           [np.sin(rotation), np.cos(rotation), 0],
                           [0, 0, 1]])
    
    # 计算新的旋转角度
    new_rotation = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    
    # 创建一个新的3D框数组
    transformed_box = np.array([transformed_center[0],
                                transformed_center[1],
                                transformed_center[2] + l/2, #label里的z值是底部的值，所以加上一半的高度
                                w,h,l,new_rotation]) #l，w交换是因为要调坐标关系
    # print(transformed_box)
    
    return transformed_box
#---------------------utils end-------------------------------

class DelftLidarDataset(base_dataset.BaseDataset):
    def __init__(self, path, split, category_name="Car", **kwargs):
        super().__init__(path, split, category_name, **kwargs)
        self.split = split
        #----------vod 数据集相关-------------------------------------
        self.DelftLocation = KittiLocations(root_dir=path,
                                output_dir="delft_output/",
                                frame_set_path="",
                                pred_dir="",
                                )
        self.origin = 'lidar' #lidar坐标系设置为参考坐标系，加载的radar数据会转化到这个坐标系下面
        #---------------vod 数据集相关 end ---------------------------------------------         

        self.scene_list = self._build_scene_list(split)
        self.velos = defaultdict(dict)
        self.calibs = {}
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        self.coordinate_mode = kwargs.get('coordinate_mode', 'velodyne')
        self.preload_offset = kwargs.get('preload_offset', -1)
        if self.preloading:
            self.training_samples = self._load_data()

    @staticmethod
    def _build_scene_list(split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                scene_names = list(range(8, 9))  # tiny 片段1，298帧
            else:
                scene_names = list(range(0, 7))   # origin
                # scene_names = list(range(7, 11)) #val 有标签
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                scene_names = list(range(8, 9))  # tiny 片段1，298帧
            else:
                scene_names = list(range(7, 11))  # origin
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                scene_names = list(range(8, 9))  # tiny 片段1，298帧
            else:
                scene_names = list(range(7, 11)) #val 有标签

        else:  # Full Dataset
            scene_names = list(range(15)) #否则是全部片段
        
        return scene_names

    def _load_data(self):
        print('preloading data into memory')
        preload_data_path = os.path.join(self.KITTI_Folder,
                                         f"preload_kitti_{self.category_name}_{self.split}_{self.coordinate_mode}_{self.preload_offset}.dat")
        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
        else:
            print('reading from annos')
            training_samples = []
            for i in range(len(self.tracklet_anno_list)):
                frames = []
                for anno in self.tracklet_anno_list[i]:
                    frames.append(self._get_frame_from_anno(anno))
                training_samples.append(frames)
            with open(preload_data_path, 'wb') as f:
                print(f'saving loaded data to {preload_data_path}')
                pickle.dump(training_samples, f)
        return training_samples

    def get_num_scenes(self):
        return len(self.scene_list)

    def get_num_tracklets(self):
        return len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        return sum(self.tracklet_len_list)

    def get_num_frames_tracklet(self, tracklet_id):
        return self.tracklet_len_list[tracklet_id]

   #专注于判断box在不在范围里, 加上：判断box里面的点的数量，删除空box，(训练时删除空box，测试时不删除空box)
    def process_tracklets_V2(self,tracklet_df, track_id_list, x_limit, y_limit, z_limit, min_length):
        """
        根据给定的x和y限制，处理tracklets数据。

        参数:
            tracklet_df: pandas.DataFrame, 包含tracklet信息的数据帧
            track_id_list: list, 所有Tracklet的唯一id列表
            x_limit: list, x的范围，格式为[x_min, x_max]
            y_limit: list, y的范围，格式为[y_min, y_max]
            min_length: int, 最短子tracklet长度

        返回:
            new_tracklet_df: pandas.DataFrame, 处理过后的tracklet数据
            track_id_list: list, 更新后的所有Tracklet的唯一id列表
        """

        # generate a list as temporary index
        temp_index = list(range(len(tracklet_df)))

        out_range_indices = list()

        # 判断 judge if x and y are in the range
        for idx, row in tracklet_df.iterrows():
            frame = row['frame']
            box = self.get_lidarbox(row,frame)
            # #统计box信息
            # with open('./%s_all_box_stats.txt'%(str(self.class_names[0])), 'a+') as f:
            #     print(f"{box[0]},{box[1]},{box[2]}",file=f)
            corners = boxes_to_corners_3d(box.reshape(-1,7))
            points = self.get_lidar(frame)
            in_box_num = np.count_nonzero(in_hull(points,corners.reshape(-1,3)))
            if not x_limit[0] <= box[0] <= x_limit[1] or not y_limit[0] <= box[1] <= y_limit[1] or not z_limit[0] <= box[2] <= z_limit[1] or in_box_num < 5:  # 点数小于1，不参与训练
                out_range_indices.append(idx) 
                # if platform_utils.USE_COMPUTER & True:
                #     vt.show_scenes(
                #         raw_sphere=corners.reshape(-1,3),
                #         pred_bbox=box.reshape(-1,7),
                #         pointcloud = [points],
                #         )



        # 移除不在范围之内box对应的id，remove out range indices from temp_index
        temp_index = [idx for idx in temp_index if idx not in out_range_indices]
        if not temp_index:
            return pd.DataFrame(), track_id_list  # 返回空表

        # Step3: 遍历剩余帧，为满足条件的子tracklet分配新的track_id
        sub_tracklet_list = []
        max_track_id = max(track_id_list)
        for i in range(len(temp_index) - 1):
            sub_tracklet_list.append(temp_index[i])
            if abs(temp_index[i] - temp_index[i+1]) > 1: #判断剩下的那些id是否连续，不连续需要生成新的track_id
                if len(sub_tracklet_list) >= min_length:
                    max_track_id += 1
                    tracklet_df.loc[sub_tracklet_list, 'track_id'] = max_track_id #用新ID替换原来的ID
                    track_id_list.append(max_track_id) #如果生成了新的ID，就要加到整体id列表记录下来，下次还要用
                else:
                    out_range_indices.extend(sub_tracklet_list) #不足长度的删掉，不应该再出现在表里
                sub_tracklet_list = []
        
        # 对最后一组进行处理，因为
        if temp_index:
            sub_tracklet_list.append(temp_index[-1])
            if len(sub_tracklet_list) >= min_length:
                max_track_id += 1
                tracklet_df.loc[sub_tracklet_list, 'track_id'] = max_track_id
                track_id_list.append(max_track_id)
            else:
                out_range_indices.extend(sub_tracklet_list) #不足长度的删掉，不应该再出现在表里

        # Step4: 删除out_range_indices对应的行
        new_tracklet_df = tracklet_df.drop(out_range_indices)

        return new_tracklet_df, track_id_list


    def _build_tracklet_anno(self):

        self.start_indices = []
        self.end_indices = []

        full_dir =  os.path.join(self.DelftLocation.imagesets_dir, 'full.txt')

        with open(full_dir, 'r') as f:
            lines = f.readlines()
            frame_numbers = [int(line.strip()) for line in lines]

        self.start_indices.append(frame_numbers[0])

        for i in range(len(frame_numbers) - 1):
            diff = abs(frame_numbers[i + 1] - frame_numbers[i])
            if diff > 1:
                self.end_indices.append(frame_numbers[i])
                self.start_indices.append(frame_numbers[i + 1])

        self.end_indices.append(frame_numbers[-1])

        assert len(self.start_indices) == len(self.end_indices)

        #------------获取每一个片段的起始帧和结束帧序号 end -----------------------

        #------根据用户自己定义的片段，读取label----------------------------------
        used_sequence_start = list(map(lambda i: self.start_indices[i], self.scene_list)) 
        used_sequence_end   = list(map(lambda i: self.end_indices[i], self.scene_list)) 
        assert len(used_sequence_start) == len(used_sequence_end)

        #  遍历每一个起始帧和结束帧读取label，形成每一个sequence的信息
        list_of_tracklet_anno = []
        self.first_frame_index = [0]  # 第一帧的index，这个index不是在原始数据集里的index
        # 把所有的轨迹的tracklet放在一起，记录的是这样的一种index，假如一共1000个label，前500帧是轨迹A，后500帧是轨迹B，那么first_frame_index为[0,500]
        self.length_per_tracklet = []
        number = 0
        all_track_id = list()
        for (sequence,start, end) in zip(self.scene_list,used_sequence_start, used_sequence_end):
            # 遍历一个片断(sequence)的所有label txt
            sequence_frame_label = pd.DataFrame()
            for onetxt_id in range(start,end+1):
                one_label_name = f"{onetxt_id:05d}.txt"
                one_label_dir = os.path.join(self.DelftLocation.label_dir,one_label_name)
                one_frame_label = pd.read_csv(
                    one_label_dir,
                    sep=" ",
                    # index_col=False,
                    names=[
                    "type",
                    "track_id",
                    "occluded",
                    "alpha",
                    "bbox_left", "bbox_top", "bbox_right","bbox_bottom", #2d bbox
                    "height", "width", "length", "x", "y", "z", "ry", #3d bbox
                    "score",
                    ],
                )
                one_frame_label = one_frame_label[one_frame_label["type"].isin([self.category_name])] #索引出来关心的类别，此处不支持多类别！！
                one_frame_label.insert(loc=0, column="frame", value=onetxt_id)  # 插入帧的序号
                one_frame_label.insert(loc=0, column="sequence", value=sequence)  # 插入sequence的序号
                sequence_frame_label = pd.concat([sequence_frame_label, one_frame_label], ignore_index=True)
            all_track_id.extend(list(sequence_frame_label.track_id.unique()))
                
            #得到一个sequence的所有信息之后，生成每一个tracklet的信息
            for track_id in sequence_frame_label.track_id.unique():
                seq_tracklet = sequence_frame_label[sequence_frame_label["track_id"] == track_id]  # 取出某个跟踪对象的轨迹信息
                # attention! 是不是应该按照帧序号排序？
                seq_tracklet = seq_tracklet.reset_index(drop=True)  # 对每一个轨迹信息生成新的dataframe
                # seq_tracklet被破开，现在seq_tracklet里面包含多个新的id了
                if "TRAIN" in self.split.upper() : #只有训练集作特殊筛选
                    seq_tracklet,_ = self.process_tracklets_V2(seq_tracklet,all_track_id,[-100,100],[-100,100],[-10,10],1) #1是最短序列长度
                    if len(seq_tracklet) == 0: #空表就继续
                        continue
                    for new_id in seq_tracklet.track_id.unique(): #把这些破开的小轨迹按照原来的方式走流程
                        new_sub_tracklet = seq_tracklet[seq_tracklet["track_id"] == new_id]
                        tracklet_anno = [anno for index, anno in new_sub_tracklet.iterrows()]  #  生成物体对应轨迹的list
                        list_of_tracklet_anno.append(tracklet_anno) #这个列表存储的是每一个tracklet的信息，
                        #解释：一个tracklet有多帧，这些帧形成一个列表，整个数据集有多个tracklet，所以有多个列表，这些列表就存在list_of_tracklet_anno
                        number += len(tracklet_anno)  # 每条轨迹对应帧index的统计        
                        self.first_frame_index.append(number)
                        self.length_per_tracklet.append(len(tracklet_anno))
                else: #val和test不作筛选
                    tracklet_anno = [anno for index, anno in seq_tracklet.iterrows()]  #  生成物体对应轨迹的list
                    list_of_tracklet_anno.append(tracklet_anno) #这个列表存储的是每一个tracklet的信息，
                    #解释：一个tracklet有多帧，这些帧形成一个列表，整个数据集有多个tracklet，所以有多个列表，这些列表就存在list_of_tracklet_anno
                    number += len(tracklet_anno)  # 每条轨迹对应帧index的统计
                    self.first_frame_index.append(number)
                    self.length_per_tracklet.append(len(tracklet_anno))    

        list_of_tracklet_len = self.length_per_tracklet
        assert len(list_of_tracklet_len) == len(self.length_per_tracklet), "Error: Lengths of self.first_frame_index and self.length_per_tracklet do not match."
        print(f"sample num:{sum(self.length_per_tracklet)}")
        return list_of_tracklet_anno, list_of_tracklet_len


    def get_frames(self, seq_id, frame_ids): #这个frame_ids不是全局的frame_id,是每个序列都配备的从0到len(序列)的数
        if self.preloading:
            frames = [self.training_samples[seq_id][f_id] for f_id in frame_ids]
        else:
            seq_annos = self.tracklet_anno_list[seq_id]
            frames = [self._get_frame_from_anno(seq_annos[f_id]) for f_id in frame_ids] #这里认为frame_ids是每一个tracklet独有的，每个序列都是从0开始

        return frames

    def get_lidar(self,frame_id):
        img_name = f"{frame_id:05d}"
        frame_loader = FrameDataLoader(kitti_locations=self.DelftLocation,
                             frame_number=img_name)
        frame_transforms = FrameTransformMatrix(frame_loader)

        if self.origin == 'camera':
            transform_matrices = {
                'camera': np.eye(4, dtype=float),
                'lidar': frame_transforms.t_camera_lidar,
                'radar': frame_transforms.t_camera_radar
            }
        elif self.origin == 'lidar':
            transform_matrices = {
                'camera': frame_transforms.t_lidar_camera,
                'lidar': np.eye(4, dtype=float),
                'radar': frame_transforms.t_lidar_radar
            }
        elif self.origin == 'radar':
            transform_matrices = {
                'camera': frame_transforms.t_radar_camera,
                'lidar': frame_transforms.t_radar_lidar,
                'radar': np.eye(4, dtype=float)
            }
        else:
            raise ValueError("Origin must be camera, lidar or radar!")
        
        points_in_origin = transform_pcl_with_all_feature(points=frame_loader.lidar_data,
                                                  transform_matrix=transform_matrices['lidar'])
        
        return points_in_origin

    def get_lidarbox(self,label: pd.Series,frame) -> np.array:
        # 解压数据
        x = label['x']
        y = label['y']
        z = label['z']
        height = label['height'] #坐标错位
        width = label['width'] #坐标错位
        length = label['length'] #坐标错位
        ry = label['ry']

        box = np.array([x,y,z,height,width,length,ry])

        # 加载变换矩阵
        frame_loader = FrameDataLoader(kitti_locations=self.DelftLocation,
                                       frame_number=f"{frame:05d}")
        frame_transforms = FrameTransformMatrix(frame_loader)  # 用于处理label

        if self.origin == 'camera':
            transform_matrices = {
                'camera': np.eye(4, dtype=float),
                'lidar': frame_transforms.t_camera_lidar,
                'radar': frame_transforms.t_camera_radar
            }
        elif self.origin == 'lidar':
            transform_matrices = {
                'camera': frame_transforms.t_lidar_camera,
                'lidar': np.eye(4, dtype=float),
                'radar': frame_transforms.t_lidar_radar
            }
        elif self.origin == 'radar':
            transform_matrices = {
                'camera': frame_transforms.t_radar_camera,
                'lidar': frame_transforms.t_radar_lidar,
                'radar': np.eye(4, dtype=float)
            }
        else:
            raise ValueError("Origin must be camera, lidar or radar!")

       

        return transform_3dbox(box,transform_matrices["camera"])

    def _get_frame_from_anno(self, anno):
        frame = anno['frame']
        box = self.get_lidarbox(anno,frame)
        center = box[:3]
        dx,dy,dz = box[3:6]
        size = [dy,dx,dz]
        ry = -box[6] 

        pc = PointCloud(self.get_lidar(frame).reshape(-1, 4).T)
        orientation = Quaternion(
                axis=[0, 0, -1], radians=ry)

        bb = Box(center, size, orientation)
        # pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)
      
        return {"pc": pc, "3d_bbox": bb, 'meta': anno}
