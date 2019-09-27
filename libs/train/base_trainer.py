import torch
from torch.utils import data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim as optim

from libs.spatial_transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, Scale, CenterCrop
from libs.temporal_transforms import TemporalRandomCrop, LoopPadding
from libs.target_transforms import ClassLabel
from libs.datasets.dataset_factory import get_training_set
from libs.spatial_transforms import MultiScaleCornerCrop
from libs.train.loss import Loss


class BaseTrainer(object):
    def __init__(self, opt, teacher, student, callback):
        self.teacher = teacher
        self.student = student
        self.callback = callback
        self.temporal_pooling = opt.train.temporal_pooling
        self.device = torch.device('cuda' if next(teacher.parameters()).is_cuda else 'cpu')
        self.opt = opt

        spatial_transform = Compose([
            MultiScaleCornerCrop(opt.train.data.scales, opt.train.data.sample_size),
            RandomHorizontalFlip(),
            ToTensor(opt.train.data.norm_value),
            Normalize(opt.train.data.mean, opt.train.data.std)
        ])
        temporal_transform = TemporalRandomCrop(opt.train.data.sample_duration)
        target_transform = ClassLabel()

        training_data = get_training_set(opt.train, spatial_transform, temporal_transform, target_transform)
        self.train_loader = data.DataLoader(training_data,
                                            batch_size=opt.train.batch_size,
                                            shuffle=True,
                                            num_workers=opt.train.n_threads,
                                            pin_memory=True)

    def _define_optimizer(self, parameters, opt):
        if opt.name == 'SGD':
            return torch.optim.SGD(parameters,
                                   lr=opt.learning_rate,
                                   momentum=opt.momentum,
                                   dampening=(opt.dampening, 0)[opt.nesterov],
                                   weight_decay=opt.weight_decay,
                                   nesterov=opt.nesterov)
        elif opt.name == 'Adam':
            return optim.Adam(parameters,
                              lr=opt.learning_rate,
                              betas=opt.get('betas', (0.9, 0.999)),
                              eps=opt.get('eps', 1e-8),
                              weight_decay=opt.weight_decay,
                              amsgrad=opt.get('amsgrad', False))
        elif opt.name == 'RMSProp':
            return optim.RMSprop(parameters,
                                 lr=opt.learning_rate,
                                 alpha=opt.get('alpha', 0.99),
                                 eps=opt.get('eps', 1e-8),
                                 weight_decay=opt.weight_decay,
                                 momentum=opt.momentum,
                                 centered=opt.get('centered', False))
        else:
            raise ValueError("There's no optimizer named '{}'!".format(opt.name))

    def _define_scheduler(self, optimizer, opt):
        if opt.name == 'ReduceLROnPlateau':
            scheduler = lrs.ReduceLROnPlateau(optimizer, mode=opt.get('mode', 'max'),
                                              factor=opt.decay_factor,
                                              patience=opt.patience,
                                              verbose=False,
                                              threshold=opt.threshold,
                                              threshold_mode=opt.get('threshold_mode', 'rel'),
                                              cooldown=opt.get('cooldown', 0),
                                              min_lr=opt.get('min_lr', 0),
                                              eps=opt.get('eps', 1e-8))
            return lambda epoch, metric: scheduler.step(metric, epoch)
        elif opt.name == 'MultiStep':
            scheduler = lrs.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
            return lambda epoch, metric: scheduler.step(epoch)
        elif opt.name == 'Exponential':
            scheduler = lrs.ExponentialLR(optimizer, gamma=opt.gamma)
            return lambda epoch, metric: scheduler.step(epoch)
        else:
            raise ValueError("There's no scheduler named '{}'!".format(opt.name))

    def _define_loss(self, opt, logit_target=True):
        loss = Loss(opt.name, opt.weight)
        target_transform = lambda x: x
        if logit_target and opt.name == 'CE':
            target_transform = F.softmax
        elif logit_target and opt.name == 'BCE':
            target_transform = F.sigmoid

        def calc_loss(outputs, targets):
            targets = target_transform(targets)
            return loss(outputs, targets)

        return calc_loss

    def _teacher_inference(self, inputs):
        if self.temporal_pooling == 'AVG':
            num_frames = inputs.size(2)
            answer_feats = []
            answer_logits = []
            for sample in inputs[:]:
                feats = []
                logits = []
                for i in range(num_frames):
                    frame = sample[:3, i, :, :].unsqueeze(0)
                    out = self.teacher(frame)
                    feats.append(out['features'])
                    logits.append(out['logits'])

                answer_feats.append(torch.sum(torch.cat(feats), dim=0) / num_frames)
                answer_logits.append(torch.sum(torch.cat(logits), dim=0) / num_frames)

            answer_feats = torch.stack(answer_feats, dim=0)
            answer_logits = torch.stack(answer_logits, dim=0)

            return {'features': answer_feats, 'logits': answer_logits}

    def get_snapshot(self):
        pass

    def train(self, epoch, step):
        pass

    def step_scheduler(self, epoch, metric):
        pass
