from libs.datasets.kinetics import Kinetics
from libs.datasets.activitynet import ActivityNet
from libs.datasets.ucf101 import UCF101
from libs.datasets.hmdb51 import HMDB51


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.data.name in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    if opt.data.name == 'kinetics':
        training_data = Kinetics(
            opt.data.video_path,
            opt.data.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.data.name == 'activitynet':
        training_data = ActivityNet(
            opt.data.video_path,
            opt.data.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.data.name == 'ucf101':
        training_data = UCF101(
            opt.data.video_path,
            opt.data.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.data.name == 'hmdb51':
        training_data = HMDB51(
            opt.data.video_path,
            opt.data.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.data.name in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    if opt.data.name == 'kinetics':
        validation_data = Kinetics(
            opt.data.video_path,
            opt.data.annotation_path,
            'validation',
            n_samples_for_each_video=opt.data.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.data.sample_duration)
    elif opt.data.name == 'activitynet':
        validation_data = ActivityNet(
            opt.data.video_path,
            opt.data.annotation_path,
            'validation',
            False,
            n_samples_for_each_video=opt.data.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.data.sample_duration)
    elif opt.data.name == 'ucf101':
        validation_data = UCF101(
            opt.data.video_path,
            opt.data.annotation_path,
            'validation',
            n_samples_for_each_video=opt.data.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.data.sample_duration)
    elif opt.data.name == 'hmdb51':
        validation_data = HMDB51(
            opt.data.video_path,
            opt.data.annotation_path,
            'validation',
            n_samples_for_each_video=opt.data.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.data.sample_duration)

    return validation_data
