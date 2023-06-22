import torch

def create_corner_timestamps(B, H, corner_num=8):
    N = (H + 1) * corner_num
    timestamps = torch.zeros((B, N, 1))

    for i in range(H):
        timestamps[:, (i * corner_num):(i * corner_num) + corner_num] = -(i + 1) * 0.1

    # 设置当前box的时间戳为0.1
    timestamps[:, -corner_num:] = 0.1

    return timestamps

B = 2  # 示例 batch 大小
H = 3  # 示例历史 box 数量
corner_num = 8  # 可选参数，默认为8

timestamps = create_corner_timestamps(B, H, corner_num)
print(timestamps)