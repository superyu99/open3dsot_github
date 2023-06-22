import open3d as o3d
import numpy as np
from math import cos, sin
import time
import os
from scipy.interpolate import CubicSpline
import seaborn as sns


def translate_boxes_to_o3d_instance(gt_boxes,color):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    # lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)

    return line_set
import numpy as np
import open3d as o3d

def create_cylinder_mesh(point1, point2, radius, resolution=20,color=[1,0,0]):
    """
    Create a cylinder that connects point1 and point2.
    The cylinder has a specified radius and resolution.
    """
    # Calculate the direction vector and length of the cylinder
    direction = np.array(point2) - np.array(point1)
    length = np.linalg.norm(direction)
    direction = direction / length

    # Create a rotation matrix that rotates the z-axis onto the direction vector
    axis = np.cross([0, 0, 1], direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot([0, 0, 1], direction))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # Create the cylinder
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, length, resolution)
    cylinder = cylinder.rotate(rotation_matrix, center=[0, 0, 0])
    cylinder = cylinder.translate((point1 + point2) / 2)

    cylinder.vertex_colors = o3d.utility.Vector3dVector([color for _ in range(len(cylinder.vertices))])

    return cylinder



#适用于MM_track,把8*3的点转化为lineset box：
def translate_M2corners_to_o3d_instance(corners,color=None):
    #corners 8*3
    lines = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    ]

    # Create a LineSet from the vertices and lines
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    mesh_cylinders = o3d.geometry.TriangleMesh()
    lines = np.asarray(line_set.lines)
    points = np.asarray(line_set.points)

    for line in lines:
        point1 = points[line[0]]
        point2 = points[line[1]]
        cylinder = create_cylinder_mesh(point1, point2, radius=0.02,color=color)
        mesh_cylinders += cylinder
   
    # if color is not None:
    #     line_set.paint_uniform_color(color)

    return mesh_cylinders



#适用于delft，但是写的有问题，因为长宽高理解错了
def create_3d_bounding_box(box, color=None): 
    center, width, height, length, rotation = box[:3], box[3], box[4], box[5], box[6]
    
    # 计算旋转矩阵
    R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                  [np.sin(rotation),  np.cos(rotation), 0],
                  [0,                0,                1]])

    # 计算8个顶点的位置
    half_dims = np.array([length, height, width]) / 2
    offsets = np.array([[1, 1, 1],
                        [1, 1, -1],
                        [1, -1, 1],
                        [1, -1, -1],
                        [-1, 1, 1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [-1, -1, -1]])
    vertices = np.dot(offsets * half_dims, R.T) + center

    # 创建线段列表，表示bounding box的12条边
    lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    # 创建Open3D的LineSet对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if color is not None:
        line_set.paint_uniform_color(color)

    return line_set





def draw_cylinder_vectors(vectors, body_width, body_color):
    cylinders = []
    for vector in vectors:
        start_point = vector[0].astype(float)
        end_point = vector[1].astype(float)
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        direction /= length

        # 创建箭头主体（圆柱）
        mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=body_width, height=length)
        mesh_cylinder.paint_uniform_color(body_color)
        mesh_cylinder.compute_vertex_normals()

        # 旋转箭头
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.arccos(direction[2])))

        mesh_cylinder.rotate(rotation_matrix, center=(0, 0, 0))

        # 平移箭头
        translation_matrix_body = (end_point + start_point) / 2

        mesh_cylinder.translate(translation_matrix_body)

        cylinders.append(mesh_cylinder)

    return cylinders
def add_vectors(vectors,color,vis):
    draw_cylinder_vectors(vectors,0.05,color)
    vis.add_geometry(draw_cylinder_vectors)

def add_bounding_boxes(boxes,color,vis):
    """
    展示3D bounding boxes
    color: autumn(红色) summer(绿色) spring(紫色) winter(蓝色)
    """

    color_list = sns.color_palette(color,len(boxes))

    # 创建每个3D bounding box的线框并添加到Open3D的可视化窗口中
    for i,box in enumerate(boxes):
        bbox = translate_M2corners_to_o3d_instance(box,color_list[i])
        vis.add_geometry(bbox)

def add_boxes(boxes,color,vis):
    # 创建每个3D bounding box的线框并添加到Open3D的可视化窗口中
    for i,box in enumerate(boxes):
        bbox = translate_M2corners_to_o3d_instance(box,color)
        vis.add_geometry(bbox)

    # 设置可视化窗口的初始视角和渲染器

def add_pointcloud(points,vis):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    vis.add_geometry(pcd)

def add_color_pointcloud(points, color, vis): 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # 设置点云颜色
    num_points = points.shape[0]
    colors = np.tile(np.array(color), (num_points, 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis.add_geometry(pcd)
#points：ndarray,N*3
import numpy as np
import open3d as o3d

def add_sphere(points, color, vis):
    # 创建一个球体模型的函数，输入为球体中心和颜色
    def create_sphere(center, color, radius=0.1):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        return sphere

    for point in points[:,0:3]:
        sphere = create_sphere(point, color)
        vis.add_geometry(sphere)

# points: ndarray, N*3
def add_width_sphere(points,color, vis):
    radius = 0.1
    color_list = sns.color_palette(color,points.shape[0])
    cylinder_radius = 0.05
    cylinder_color = (0, 0, 1)

    # 将点按x坐标排序
    points = sorted(points, key=lambda point: point[0])
    # 从点列表中创建NumPy数组
    points_np = np.array([point for point in points])
    # 使用CubicSpline插值平滑曲线
    cs = CubicSpline(points_np[:, 0], points_np[:, 1:])
    xs = np.linspace(points_np[0, 0], points_np[-1, 0], 100)
    interpolated_points = np.column_stack((xs, cs(xs)))

    for i,point in enumerate(points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color_list[i])
        vis.add_geometry(sphere)

    # 用 Cylinder 类模拟粗线条，并连接排序后的点
    for i in range(len(interpolated_points) - 1):
        p1 = interpolated_points[i]
        p2 = interpolated_points[i + 1]
        cylinder_length = np.linalg.norm(np.array(p2) - np.array(p1))
        cylinder_center = (np.array(p1) + np.array(p2)) / 2
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(cylinder_radius, cylinder_length)
        cylinder.compute_vertex_normals()
        cylinder.translate(-cylinder.get_center())
        cylinder.translate(cylinder_center)

        # 计算从Z轴到p2-p1向量的旋转矩阵
        z_axis = np.array([0, 0, 1])
        direction = (np.array(p2) - np.array(p1)) / cylinder_length
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        cylinder.rotate(rotation_matrix)

        cylinder.paint_uniform_color(cylinder_color)

        # 将粗线条添加到可视化器
        vis.add_geometry(cylinder)

def get_radar_velocity_vectors(pc_radar, compensated_radial_velocity):
    radial_unit_vectors = pc_radar / np.linalg.norm(pc_radar, axis=1, keepdims=True)
    velocity_vectors = compensated_radial_velocity[:, None] * radial_unit_vectors

    return velocity_vectors

def add_radar_radial_velocity(radar_pointcloud,color,vis):
    compensated_radial_velocity = radar_pointcloud[:, 5]
    pc_radar = radar_pointcloud[:, 0:3]
    velocity_vectors = get_radar_velocity_vectors(pc_radar, compensated_radial_velocity)

    end_points = pc_radar + velocity_vectors

    # Create open3d LineSet for velocity vectors
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.vstack((pc_radar, end_points)))
    lines.lines = o3d.utility.Vector2iVector(np.column_stack((np.arange(pc_radar.shape[0]), np.arange(pc_radar.shape[0], 2 * pc_radar.shape[0]))))
    lines.colors = o3d.utility.Vector3dVector(np.tile(color, (pc_radar.shape[0], 1)).astype(np.float64))

    vis.add_geometry(lines)

    

def show_scenes(bboxes=None,
                pred_bbox=None,
                gt_bbox=None,
                future_box=None,
                pointcloud=None,  # 支持多帧放在一个列表里传入
                hist_pointcloud=None,  # 支持多帧放在一个列表里传入
                future_pointcloud=None,  # 支持多帧放在一个列表里传入
                sphere=None,
                vectors=None,
                raw_sphere=None,
                red_bbox=None,
                title=None,
                radar_pointcloud=None):
    vis = o3d.visualization.Visualizer()
    
    if title is not None:
        vis.create_window(window_name=title)
    else:
        vis.create_window()

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    vis.add_geometry(mesh)

    if bboxes is not None:
        add_bounding_boxes(bboxes,"autumn",vis)
    if pred_bbox is not None:
        add_bounding_boxes(pred_bbox,"summer",vis)
    if gt_bbox is not None:
        add_bounding_boxes(gt_bbox,"winter",vis)
    if future_box is not None:
        add_bounding_boxes(future_box,"viridis",vis)
    if pointcloud is not None: #点云显示成红色
        for i,cloud in enumerate(pointcloud):
            add_color_pointcloud(cloud,[1,0,0],vis)
    if hist_pointcloud is not None:
        color_list = sns.color_palette("winter",len(hist_pointcloud))
        for i,cloud in enumerate(hist_pointcloud):
            add_color_pointcloud(cloud,color_list[i],vis)
    if future_pointcloud is not None:
        color_list = sns.color_palette("autumn",len(future_pointcloud))
        for i,cloud in enumerate(future_pointcloud):
            add_color_pointcloud(cloud,color_list[i],vis)        
    if raw_sphere is not None:
        add_sphere(raw_sphere, [1,0,0], vis)
    if sphere is not None:
        add_width_sphere(sphere,"autumn",vis)
    if vectors is not None:
        add_vectors(vectors,[0,0,1],vis)
    if red_bbox is not None:
        add_boxes(red_bbox,[1,0,0],vis)

    if radar_pointcloud is not None:
        for i,cloud in enumerate(radar_pointcloud):
            add_sphere(radar_pointcloud[i],sns.color_palette(["red"])[0],vis)
            add_radar_radial_velocity(radar_pointcloud[i],sns.color_palette(["red"])[0],vis)

    #视角控制
    camera_params_file = "sttracker-mrt-view.json"
    if camera_params_file is not None and os.path.exists(camera_params_file):
        # 从JSON文件中读取相机参数并应用到视图控件上
        param = o3d.io.read_pinhole_camera_parameters(camera_params_file)
        vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    vis.run()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(camera_params_file, param)
    vis.destroy_window()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

def corners_2d(box):
    x, y, z, w, l, h, theta = box
    dx = w / 2
    dy = l / 2

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])

    corners = np.array([[-dx, -dy],
                        [dx, -dy],
                        [dx, dy],
                        [-dx, dy]])

    rotated_corners = np.dot(rot, corners.T).T
    return rotated_corners + [x, y]

def plot_boxes(ax, boxes):
    colors = cm.rainbow(np.linspace(0, 1, len(boxes))) # 为每个盒子生成不同的颜色
    for i, box in enumerate(boxes):
        corners = corners_2d(box)
        corners = np.vstack((corners, corners[0])) # 添加第一个角点以闭合多边形
        ax.plot(corners[:, 1], -corners[:, 0], '-', color=colors[i])

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_title('Top-down view of 3D Boxes (rotated)')
    ax.grid(True)
    ax.axis('equal')

def plot_heatmap(ax, template_bev_mask):
    sns.heatmap(template_bev_mask, ax=ax, cmap='viridis', cbar=True)
    ax.set_title('Template BEV Mask')

def plot_motion_map(ax, motion_map):
    X, Y = np.meshgrid(np.arange(motion_map.shape[1]), np.arange(motion_map.shape[0]))
    U = motion_map[:, :, 0]
    V = motion_map[:, :, 1]
    ax.quiver(X, Y, U, V, scale=20)
    ax.set_title('Motion Map')

def plot_three_subplots(boxes=None, template_bev_mask=None, motion_map=None, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if boxes is not None:
        plot_boxes(axes[0], boxes)

    if template_bev_mask is not None:
        plot_heatmap(axes[1], template_bev_mask)

    if motion_map is not None:
        plot_motion_map(axes[2], motion_map)
    
    if title is not None:
        # plt.title=title
        pass
       
    plt.tight_layout()
    plt.savefig("3inone.jpg")
    

# 测试函数
if __name__ == "__main__":
    # 生成随机数据

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    bboxes = np.array([[1,1,1,1,1,1,0]])
    pred_bbox = np.array([[1,1,1,1,1,1,0.5]])
    gt_bbox = np.array([[1,1,1,1,1,1,0.7]])
    pointcloud = np.random.rand(100,3) 

    show_scenes(bboxes,pred_bbox,gt_bbox,pointcloud,vis)

    #--------------用法示例---------------------------
    # bboxes = batch_dict['history_gt_boxes'][0].detach().cpu().numpy() -> N*7
    # pred_bbox = np.concatenate((center.cpu().numpy(),gt_box[3:7].cpu().numpy())).reshape(1,7) -> N*7
    # gt_bbox = gt_box.detach().cpu().numpy().reshape(1,7) -> N*7
    # pointcloud =  batch_dict['template_points'].cpu()[:,1:4] -> N*3
    # vt.show_scenes(bboxes,pred_bbox,gt_bbox,pointcloud)
