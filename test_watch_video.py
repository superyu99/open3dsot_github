import numpy as np
import open3d as o3d
import time
from datasets import get_dataset
import yaml
from easydict import EasyDict
import vis_tool as vt
import argparse
import seaborn as sns
import os

#---------------------读取记录-------------------------------------
# 定义一个列表来保存读取的数据
data_list = []

# 打开文件并按行读取内容
with open('./track_result_radar_alltest_{}.txt'.format("Car"), 'r') as f:
    for line in f.readlines():
        # 使用逗号分隔的值将行分割为列表
        values = [float(x) for x in line.strip().split(', ')]
        
        # 提取 frame_id
        frame_id = int(values[0])
        
        # 提取预测 box 和真实 box 的坐标值
        pred_box_coords = np.array(values[1:25]).reshape(8, 3)
        gt_box_coords = np.array(values[25:]).reshape(8, 3)
        
        # 将 frame_id、预测 box 和真实 box 添加到 data_list
        data_list.append((frame_id, pred_box_coords, gt_box_coords))

# 现在，data_list 包含从文件中读取的所有 frame_id、预测 box 和真实 box
#---------------------读取记录 end ------------------------------------



# ----------------准备数据集-------------------------------------------
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
    parser.add_argument('--cfg', type=str, default="cfgs/M2_track_delft_radar.yaml",help='the config_file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)
#--------------准备数据集----------------------------------------------

def get_dynamic_visualizer():
    #创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().point_size = 5  # set points size

    def save_view(vis):
        # get camera parameters
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("view_delft_M2.json", params)

    vis.register_animation_callback(save_view)

    return vis

def show_pointcloud(vis,index):
    
    start = time.time()
    data = data_list[index]
    frame = data[0]
    predbox = data[1]
    gtbox = data[2]

    radar = test_data.dataset.get_radar(frame)
    lidar = test_data.dataset.get_lidar(frame)


    vis.clear_geometries()
   # 设置视角缩放比例

    vt.add_boxes([predbox],[1,0,0],vis)
    vt.add_boxes([gtbox],[0,0,1],vis)
    vt.add_color_pointcloud(radar,sns.color_palette(["red"])[0],vis)
    vt.add_color_pointcloud(lidar,sns.color_palette(["Purple"])[0],vis)
    
    # vt.add_sphere(radar, [1,0,0], vis)

    # vt.add_sphere(radar,sns.color_palette(["red"])[0],vis)
    vt.add_radar_radial_velocity(radar,sns.color_palette(["red"])[0],vis)


    #绘制原点
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

    vis.add_geometry(axis_pcd)

    print("frame:"+str(index))
    #视角控制
    camera_params_file = "view_delft_M2.json"
    if camera_params_file is not None and os.path.exists(camera_params_file):
        # 从JSON文件中读取相机参数并应用到视图控件上
        param = o3d.io.read_pinhole_camera_parameters(camera_params_file)
        vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    

    vis.poll_events()
    vis.update_renderer()

    



cfg = parse_config()
test_data = get_dataset(cfg, type='test', split='test')    

vis= get_dynamic_visualizer()


for i in range(len(data_list)):
    # 显示图像并保存
    show_pointcloud(vis,i)
    time.sleep(0.1)

    
    # vis.run()

