"""
___init__.py
Created by zenn at 2021/7/18 15:50
"""
from datasets import kitti, sampler, nuscenes_data, \
                    waymo_data, delft_lidar, delft_radar, \
                    delft_radar_lidar, delft_masked_image, \
                    delft_radar_vdxdy, nuscenes_lidar_radar, \
                    nuscenes_lidar_mf

def get_dataset(config, type='train', **kwargs):
    if config.dataset == 'kitti':
        data = kitti.kittiDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)
    elif config.dataset == 'delft_lidar':
        data = delft_lidar.DelftLidarDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)

    elif config.dataset == 'delft_radar':
        data = delft_radar.DelftRadarDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)
    elif config.dataset == 'delft_image':
        data = delft_masked_image.DelftImageDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)

    elif config.dataset == 'delft_radar_lidar':
        data = delft_radar_lidar.DelftRadarLidarDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)

    elif config.dataset == 'delft_radar_vdxdy':
        data = delft_radar_vdxdy.DelftRadarDatasetVdxdy(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)

    elif config.dataset == 'nuscenes':
        data = nuscenes_data.NuScenesDataset(path=config.path,
                                             split=kwargs.get('split', 'train_track'),
                                             category_name=config.category_name,
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             # can only use keyframes for training
                                             preloading=config.preloading,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             min_points=1 if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1)
    elif config.dataset == 'nuscenes_mf':
        data = nuscenes_lidar_mf.NuScenesMFDataset(path=config.path,
                                             split=kwargs.get('split', 'train_track'),
                                             category_name=config.category_name,
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             # can only use keyframes for training
                                             preloading=config.preloading,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             min_points=1 if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1,
                                            hist_num = config.hist_num)
    elif config.dataset == 'nuscenes_lidar_radar':
        data = nuscenes_lidar_radar.NuScenesLidarRadarDataset(path=config.path,
                                             split=kwargs.get('split', 'train_track'),
                                             category_name=config.category_name,
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             # can only use keyframes for training
                                             preloading=config.preloading,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             min_points=1 if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1)

    elif config.dataset == 'waymo':
        data = waymo_data.WaymoDataset(path=config.path,
                                       split=kwargs.get('split', 'train'),
                                       category_name=config.category_name,
                                       preloading=config.preloading,
                                       preload_offset=config.preload_offset,
                                       tiny=config.tiny)
    else:
        data = None

    if type == 'train_siamese':
        return sampler.PointTrackingSampler(dataset=data,
                                            random_sample=config.random_sample,
                                            sample_per_epoch=config.sample_per_epoch,
                                            config=config)
    elif type.lower() == 'train_motion':
        return sampler.MotionTrackingSampler(dataset=data,
                                             config=config)
    elif type.lower() == 'train_motion_mf':
        return sampler.MotionTrackingSamplerMF(dataset=data,
                                             config=config)
    elif type.lower() == 'train_motion_cross_modal':
        return sampler.MotionTrackingSamplerRadarLidar(dataset=data,
                                             config=config)
    elif type.lower() == 'train_motion_image':
        return sampler.MotionTrackingSamplerImage(dataset=data,
                                             config=config)
    else:
        return sampler.TestTrackingSampler(dataset=data, config=config)
