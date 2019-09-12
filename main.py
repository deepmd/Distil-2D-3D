import torch
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import json
import random
import os
from libs.utils import get_logger, count_parameter_number
from libs.spatial_transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, Scale, CenterCrop
from libs.temporal_transforms import TemporalRandomCrop, LoopPadding
from libs.target_transforms import ClassLabel
from libs.dataset import get_training_set, get_validation_set
from libs.spatial_transforms import MultiScaleCornerCrop

from models.model_factory import get_models

from libs.opts import opts


def teacher_inference(model, input, temporal_pooling):
    if temporal_pooling == 'avg':
        num_frames = input.size(2)
        answer_feats = []
        answer_logits = []
        for sample in input[:]:
            feats = []
            logits = []
            for i in range(num_frames):
                frame = sample[:3, i, :, :].unsqueeze(0)
                out = model(frame)
                feats.append(out['features'])
                logits.append(out['logits'])

            answer_feats.append(torch.sum(torch.cat(feats), dim=0) / num_frames)
            answer_logits.append(torch.sum(torch.cat(logits), dim=0) / num_frames)

        answer_feats = torch.stack(answer_feats, dim=0)
        answer_logits = torch.stack(answer_logits, dim=0)

        return {'features': answer_feats, 'logits': answer_logits}

def train(opt, writer, logger):
    # Setup seeds
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    random.seed(opt.manual_seed)

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    cudnn.benchmark = True

    # Setup Models
    teacher, student = get_models(opt.t_arch, opt.s_arch,
                                  opt.n_outputs,
                                  opt.t_pretrain_path,
                                  opt.freeze_t_bbon)
    teacher = teacher.to(device)
    student = student.to(device)

    # count parameter number
    print('Teacher model:')
    count_parameter_number(teacher)

    print('Student model:')
    count_parameter_number(student)

    if not opt.no_train:
        spatial_transform = Compose([
            MultiScaleCornerCrop(opt.scales, opt.sample_size),
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), 
            Normalize(opt.mean, opt.std)
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()

        training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
        train_loader = data.DataLoader(training_data,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.n_threads,
                                       pin_memory=True)

    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value),
            Normalize(opt.mean, opt.std)
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
        val_loader = data.DataLoader(validation_data,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     num_workers=opt.n_threads,
                                     pin_memory=True)

    optimizer = optim.SGD(student.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          dampening=opt.dampening,
                          weight_decay=opt.weight_decay,
                          nesterov=opt.nesterov)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

    for i, (inputs, targets) in enumerate(train_loader):
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs.to(device))
        # targets = Variable(targets)

        student.train()
        teacher.eval()

        answer = teacher_inference(teacher, inputs, opt.temporal_pooling)

        s_inputs = F.interpolate(inputs, scale_factor=(1.0, 0.5, 0.5), mode='nearest')
        output = student(s_inputs)


if __name__ == '__main__':
    opt = opts().parse()
    print(opt)

    writer = SummaryWriter(log_dir=opt.save_path)

    print("RUNDIR: {}".format(opt.save_path))
    with open(os.path.join(opt.save_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    logger = get_logger(opt.save_path)
    logger.info("Let the games begin")

    train(opt, writer, logger)
