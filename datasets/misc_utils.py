import numpy as np
#-----------------------------------------------------------------
#获取历史帧的id并伴随是否是真实的历史帧id的mask
def get_history_frame_ids_and_masks(this_frame_id, hist_num):
    history_frame_ids = []
    masks = []
    for i in range(1, hist_num + 1):
        frame_id = this_frame_id - i
        if frame_id < 0:
            frame_id = 0
            masks.append(0)
        else:
            masks.append(1)
        history_frame_ids.append(frame_id)
    return history_frame_ids, masks

# # 测试代码
# this_frame_id = 13
# hist_num = 3
# history_frame_ids, masks = get_history_frame_ids_and_masks(this_frame_id, hist_num)
# print(history_frame_ids)  # 输出：[12, 11, 10]
# print(masks)  # 输出：[1, 1, 1]
# def test_get_history_frame_ids_and_masks():
#     # 测试用例 1
#     this_frame_id = 1
#     hist_num = 3
#     expected_history_frame_ids = [0, 0, 0]
#     expected_masks = [1, 0, 0]
#     output_history_frame_ids, output_masks = get_history_frame_ids_and_masks(this_frame_id, hist_num)
#     assert output_history_frame_ids == expected_history_frame_ids
#     assert output_masks == expected_masks

#     # 测试用例 2
#     this_frame_id = 2
#     hist_num = 5
#     expected_history_frame_ids = [1, 0, 0, 0, 0]
#     expected_masks = [1, 1, 0, 0, 0]
#     output_history_frame_ids, output_masks = get_history_frame_ids_and_masks(this_frame_id, hist_num)
#     assert output_history_frame_ids == expected_history_frame_ids
#     assert output_masks == expected_masks

#     # 测试用例 3
#     this_frame_id = 0
#     hist_num = 3
#     expected_history_frame_ids = [0, 0, 0]
#     expected_masks = [0, 0, 0]
#     output_history_frame_ids, output_masks = get_history_frame_ids_and_masks(this_frame_id, hist_num)
#     assert output_history_frame_ids == expected_history_frame_ids
#     assert output_masks == expected_masks

#     print("所有测试用例均通过！")

# 运行测试
# test_get_history_frame_ids_and_masks()
#-------------------------------------------------------------------

#-------------------------------------------------------------------
#用于把多个历史帧转化成dict的函数，-1：最近的历史帧（当前帧的前一帧），-2：第二个历史帧
def create_history_frame_dict(prev_frames):
    history_frame_dict = {}
    for i, frame in enumerate(prev_frames):
        key = -1 * (i + 1)
        history_frame_dict[str(key)] = frame
    return history_frame_dict
#-------------------------------------------------------------------

#-------------------------------------------------------------------
#为历史帧生成时间戳，第一个历史帧-0.1 -0.2 ...
def generate_timestamp_prev_list(valid_mask, point_sample_size):
    timestamp_prev_list = []
    valid_time = 0

    for mask in valid_mask:
        if mask == 1:
            valid_time -= 0.1
            timestamp_prev = np.full((point_sample_size, 1), fill_value=valid_time)
        else:
            # 保持当前有效时间戳
            timestamp_prev = np.full((point_sample_size, 1), fill_value=valid_time)
        timestamp_prev_list.append(timestamp_prev)
    
    return timestamp_prev_list
#
#-------------------------------------------------------------------

#----------------------获取最后N个历史box------------------------
def get_last_n_bounding_boxes(results_bbs, mask):
    last_n_bbs = []
    last_valid_index = len(results_bbs) - 1
    for m in mask:
        if m == 1 and last_valid_index >= 0:
            last_n_bbs.append(results_bbs[last_valid_index])
            last_valid_index -= 1
        elif len(last_n_bbs) > 0:
            last_n_bbs.append(last_n_bbs[-1])
    return last_n_bbs

# # 测试用例
# results_bbs = ["box0","box1","box2"]  # 只有一个bounding box
# mask = [1, 1, 1]
# expected_result = [
#    "box2","box1","box0"
# ]

# output = get_last_n_bounding_boxes(results_bbs, mask)
# assert output == expected_result, f"Expected {expected_result}, but got {output}"
# print("测试通过!")
#
#----------------------获取最后N个历史box------------------------

#----------------------用新的refbox替换旧的--------------------
def update_results_bbs(results_bbs, valid_mask, new_refboxs):
    # 获取需要更新的元素数量
    update_count = int(sum(valid_mask))
    # 获取总的参考 box 数量
    N = len(new_refboxs)

    # 判断 results_bbs 长度
    if len(results_bbs) >= (N + 1):
        # 直接用 new_refboxs 里面的元素替换 results_bbs 的最后 N 个元素，new_refboxs顺序读，但是写入results_bbs是倒序写入
        for i in range(N):
            results_bbs[-(i+1)] = new_refboxs[i]
    else:
        # 直接用 new_refboxs 里面的 (update_count - 1) 个元素替换 results_bbs 的最后 N 个元素，new_refboxs顺序读，但是写入results_bbs是倒序写入
        for i in range(update_count-1): #这是特殊处理刚开始那几帧,减去1是为了不更新最初的那个真值
            results_bbs[-(i+1)] = new_refboxs[i]
        
    return results_bbs

# def test_update_results_bbs():
#     # 测试用例 1 ref0此时是真值不用更新，函数要能够保持result_bbs
#     result_bbs_1 = ["ref0"]
#     mask_1 = [1, 0, 0]
#     new_refboxs_1 = ["box0", "box1", "box2"]
#     assert update_results_bbs(result_bbs_1, mask_1, new_refboxs_1) == ["ref0"]

#     # 测试用例 2
#     result_bbs_2 = ["ref0", "ref1"]
#     mask_2 = [1, 1, 0]
#     new_refboxs_2 = ["box0", "box1", "box2"]
#     assert update_results_bbs(result_bbs_2, mask_2, new_refboxs_2) == ["ref0", "box0"]

#     # 测试用例 3
#     result_bbs_3 = ["ref0", "ref1", "ref3",]
#     mask_3 = [1, 1, 1]
#     new_refboxs_3 = ["box0", "box1", "box2"]
#     assert update_results_bbs(result_bbs_3, mask_3, new_refboxs_3) == ["ref0", "box1", "box0"]

#     # 测试用例 4
#     result_bbs_4 = ["ref0", "ref1", "ref3", "ref4"]
#     mask_4 = [1, 1, 1]
#     new_refboxs_4 = ["box0", "box1", "box2"]
#     assert update_results_bbs(result_bbs_4, mask_4, new_refboxs_4) == ["ref0", "box2", "box1", "box0"]

# test_update_results_bbs()
#--------------------------------------------------------------

#------------------------获取tensor版本的corners---------------------------
import torch

def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def get_tensor_corners(center,wlh,theta,wlh_factor=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.

        """
        w, l, h = wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * torch.tensor([1,  1,  1,  1, -1, -1, -1, -1], dtype=torch.float32, device=center.device)
        y_corners = w / 2 * torch.tensor([1, -1, -1,  1,  1, -1, -1,  1], dtype=torch.float32, device=center.device)
        z_corners = h / 2 * torch.tensor([1,  1, -1, -1,  1,  1, -1, -1], dtype=torch.float32, device=center.device)
        corners = corners = torch.stack((x_corners, y_corners, z_corners), dim=0)

        # Rotate
        corners = _axis_angle_rotation("Z",-theta)@corners

        # Translate
        x, y, z = center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

# #测试程序
# from datasets.data_classes import Box
# from pyquaternion import Quaternion
# import numpy as np
# orientation = Quaternion(
#                 axis=[0, 0, -1], radians=0.3)

# bb = Box(np.array([1,1,1]), np.array([1,1,1]), orientation)

# print(bb.corners().T)
# print(get_tensor_corners(torch.tensor([1,1,1]),torch.tensor([1,1,1]),torch.tensor(0.3)).T)
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# 为corners tensor生成时间戳
def create_corner_timestamps(B, H, corner_num=8):
    """
    为B*N*3的corners生成时间戳：当前帧在最后，历史帧在前面,-0.1,-0.2-0.3 ... 当前帧+0.1
    N应该等于 (历史帧数量+1)*8 
    返回的张量可以直接拼接在原始张量后面
    """
    N = (H + 1) * corner_num
    timestamps = torch.zeros((B, N, 1))

    for i in range(H):
        timestamps[:, (i * corner_num):(i * corner_num) + corner_num] = -(i + 1) * 0.1

    # 设置当前box的时间戳为0.1
    timestamps[:, -corner_num:] = 0.1

    return timestamps

# B = 2  # 示例 batch 大小
# H = 3  # 示例历史 box 数量
# corner_num = 8  # 可选参数，默认为8

# timestamps = create_corner_timestamps(B, H, corner_num)
# print(timestamps)
#---------------------------------------------------------------------------