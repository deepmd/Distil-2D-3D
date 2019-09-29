from torch.nn import functional as F

from libs.train.base_trainer import BaseTrainer
from libs.train.loss import Loss
from libs.utils import get_model_state


class SimpleTrainer(BaseTrainer):
    def __init__(self, opt, teacher, student, callback, checkpoint):
        super(SimpleTrainer, self).__init__(opt, teacher, student, callback)

        # Optimizer
        self.optimizer = self._define_optimizer(self.student.parameters(), opt.train.optimizer or opt.train.s_optimizer)
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Loss function
        T = opt.train.sim_loss.temperature
        loss = Loss(opt.train.sim_loss.name, opt.train.sim_loss.weight)
        def loss_fn(s_logits, t_logits):
            loss_value = loss(s_logits/T, t_logits/T)
            return loss_value
        self.similarity_loss = loss_fn

        # Schedulers
        self.scheduler = self._define_scheduler(self.optimizer, opt.train.scheduler or opt.train.s_scheduler)

    def get_snapshot(self):
        return {
            'teacher': get_model_state(self.teacher),
            'student': get_model_state(self.student),
            'optimizer': self.optimizer.state_dict()
        }

    def train(self, epoch, step):
        self.student.train()
        self.teacher.eval()

        self.callback.begin_epoch(epoch, step, len(self.train_loader))
        for i, (inputs, targets) in enumerate(self.train_loader):
            step += 1
            batch_size = inputs.shape[0]

            inputs = inputs.to(self.device)

            t_outputs = self._teacher_inference(inputs)

            s_inputs = F.interpolate(inputs, scale_factor=(1.0, 0.5, 0.5), mode='nearest')
            s_outputs = self.student(s_inputs)

            loss = self.similarity_loss(s_outputs['logits'], t_outputs['logits'])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {
                'Loss': loss.mean().item()
            }
            lrs = {
                'Opt': self.optimizer.param_groups[0]['lr']
            }
            self.callback.end_iter(epoch, step, losses, lrs, batch_size)

        self.callback.end_epoch(epoch, step)
        return step

    def step_scheduler(self, epoch, metric):
        self.scheduler(epoch, metric)
