import argparse
import yaml
from os import path

from .mean import get_mean, get_std
# from .spatial_transforms import MultiScaleRandomCrop, MultiScaleCornerCrop


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--exp_id',
                                 type=str,
                                 default='default')
        self.parser.add_argument(
            '--exp_path',
            default='./exps',
            type=str,
            help='Expriments directory path')
        self.parser.add_argument(
            '--train_dataset',
            default='kinetics',
            type=str,
            help='Used dataset for training (activitynet | kinetics | ucf101 | hmdb51)')
        self.parser.add_argument(
            '--val_dataset',
            default='kinetics',
            type=str,
            help='Used dataset for validatin (ucf101 | hmdb51)')
        self.parser.add_argument(
            '--n_outputs',
            default=400,
            type=int,
            help=
            'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)'
        )
        self.parser.add_argument(
            '--n_finetune_classes',
            default=400,
            type=int,
            help=
            'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
        )
        self.parser.add_argument(
            '--sample_size',
            default=224,
            type=int,
            help='Height and width of inputs')
        self.parser.add_argument(
            '--sample_duration',
            default=16,
            type=int,
            help='Temporal duration of inputs')
        self.parser.add_argument(
            '--initial_scale',
            default=1.0,
            type=float,
            help='Initial scale for multiscale cropping')
        self.parser.add_argument(
            '--n_scales',
            default=5,
            type=int,
            help='Number of scales for multiscale cropping')
        self.parser.add_argument(
            '--scale_step',
            default=0.84089641525,
            type=float,
            help='Scale step for multiscale cropping')
        self.parser.add_argument(
            '--train_crop',
            default='corner',
            type=str,
            help=
            'Spatial cropping method in training. random is uniform. '
            'corner is selection from 4 corners and 1 center.  (random | corner | center)'
        )
        self.parser.add_argument(
            '--learning_rate',
            default=0.1,
            type=float,
            help=
            'Initial learning rate (divided by 10 while training by lr scheduler)')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
        self.parser.add_argument(
            '--dampening', default=0.9, type=float, help='dampening of SGD')
        self.parser.add_argument(
            '--weight_decay', default=1e-3, type=float, help='Weight Decay')
        self.parser.add_argument(
            '--mean_dataset',
            default='activitynet',
            type=str,
            help=
            'dataset for mean values of mean subtraction (activitynet | kinetics)')
        self.parser.add_argument(
            '--no_mean_norm',
            action='store_true',
            help='If true, inputs are not normalized by mean.')
        self.parser.set_defaults(no_mean_norm=False)
        self.parser.add_argument(
            '--std_norm',
            action='store_true',
            help='If true, inputs are normalized by standard deviation.')
        self.parser.set_defaults(std_norm=False)
        self.parser.add_argument(
            '--nesterov',
            action='store_true',
            help='Nesterov momentum')
        self.parser.set_defaults(nesterov=False)
        self.parser.add_argument(
            '--optimizer',
            default='sgd',
            type=str,
            help='Currently only support SGD')
        self.parser.add_argument(
            '--lr_patience',
            default=10,
            type=int,
            help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
        )
        self.parser.add_argument(
            '--batch_size',
            default=128,
            type=int,
            help='Batch Size')
        self.parser.add_argument(
            '--n_epochs',
            default=200,
            type=int,
            help='Number of total epochs to run')
        self.parser.add_argument(
            '--begin_epoch',
            default=1,
            type=int,
            help=
            'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
        )
        self.parser.add_argument(
            '--n_val_samples',
            default=3,
            type=int,
            help='Number of validation samples for each activity')
        self.parser.add_argument(
            '--s_resume_path',
            default='',
            type=str,
            help='Save student model (.pth) of previous training')
        self.parser.add_argument(
            '--t_pretrain_path',
            default='',
            type=str,
            help='Pretrained teacher model (.pth)')
        self.parser.add_argument(
            '--ft_begin_index',
            default=0,
            type=int,
            help='Begin block index of fine-tuning')
        self.parser.add_argument(
            '--no_train',
            action='store_true',
            help='If true, training is not performed.')
        self.parser.set_defaults(no_train=False)
        self.parser.add_argument(
            '--no_val',
            action='store_true',
            help='If true, validation is not performed.')
        self.parser.set_defaults(no_val=False)
        self.parser.add_argument(
            '--test', action='store_true', help='If true, test is performed.')
        self.parser.set_defaults(test=False)
        self.parser.add_argument(
            '--test_subset',
            default='val',
            type=str,
            help='Used subset in test (val | test)')
        self.parser.add_argument(
            '--scale_in_test',
            default=1.0,
            type=float,
            help='Spatial scale in test')
        self.parser.add_argument(
            '--crop_position_in_test',
            default='c',
            type=str,
            help='Cropping method (c | tl | tr | bl | br) in test')
        self.parser.add_argument(
            '--no_softmax_in_test',
            action='store_true',
            help='If true, output for each clip is not normalized using softmax.')
        self.parser.set_defaults(no_softmax_in_test=False)
        self.parser.add_argument(
            '--no_cuda', action='store_true', help='If true, cuda is not used.')
        self.parser.set_defaults(no_cuda=False)
        self.parser.add_argument(
            '--n_threads',
            default=4,
            type=int,
            help='Number of threads for multi-thread loading')
        self.parser.add_argument(
            '--checkpoint',
            default=10,
            type=int,
            help='Trained model is saved at every this epochs.')
        self.parser.add_argument(
            '--no_hflip',
            action='store_true',
            help='If true holizontal flipping is not performed.')
        self.parser.set_defaults(no_hflip=False)
        self.parser.add_argument(
            '--norm_value',
            default=1,
            type=int,
            help=
            'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
        self.parser.add_argument(
            '--t_arch',
            default='res50',
            type=str,
            help='2D model is used as teacher (res18 | res34 | res50 | res101 | tes152).')
        self.parser.add_argument(
            '--s_arch',
            default='c3d',
            type=str,
            help='3D model is used as student (c3d | r3d | r21d).')
        self.parser.add_argument(
            '--manual_seed',
            default=1337,
            type=int,
            help='Manually set random seed')
        self.parser.add_argument(
            '--gpus',
            type=str,
            default='0',
            help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument(
            '--freeze_t_bbon',
            type=bool,
            default=True,
            help='Freeze teacher model\'s backbone ')
        self.parser.add_argument(
            '--temporal_pooling',
            type=str,
            default='avg',
            help='How to combine frame information across a video (avg | vlad | ssl)')
        self.parser.add_argument(
            '--dataset_configs',
            type=str,
            help='Configurations of datasets (kinetics | ucf101 | hmdb51).')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        with open(opt.dataset_configs) as fp:
            cfg = yaml.load(fp)

        train_data_cfg = cfg[opt.train_dataset]
        if train_data_cfg is not None:
            opt.train_video_path = path.join(train_data_cfg['data_path'], train_data_cfg['jpg_video_path'])
            opt.train_annotation_path = path.join(train_data_cfg['data_path'],
                                                  train_data_cfg['annotation_path'], 'ucf101_01.json')
            opt.save_path = path.join(opt.exp_path, opt.exp_id)
            if opt.s_resume_path:
                opt.s_resume_path = path.join(opt.save_path, opt.s_resume_path)

        val_data_cfg = cfg[opt.val_dataset]
        if val_data_cfg is not None:
            opt.val_video_path = path.join(val_data_cfg['data_path'], val_data_cfg['avi_video_path'])
            opt.val_annotation_path = path.join(val_data_cfg['data_path'], val_data_cfg['annotation_path'])
            opt.extracted_features_path = path.join(opt.save_path, 'extracted_features_dir')


        opt.scales = [opt.initial_scale]
        for i in range(1, opt.n_scales):
            opt.scales.append(opt.scales[-1] * opt.scale_step)

        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        if opt.no_mean_norm and not opt.std_norm:
            opt.mean = [0, 0, 0]
            opt.std = [1, 1, 1]
        elif not opt.std_norm:
            opt.mean = opt.mean
            opt.std = [1, 1, 1]

        if opt.nesterov:
            opt.dampening = 0

        return opt
