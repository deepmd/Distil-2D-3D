import torch
from torch.autograd import Variable
from torch.nn import functional as F

from libs.train.base_trainer import BaseTrainer
from libs.train.loss import Loss, Regularizer
from libs.models.model_factory import get_discriminator
from libs.utils import get_model_state, log_parameter_number


class ANCTrainer(BaseTrainer):
    """ Belagiannis et al., Adversarial Network Compression
        (http://arxiv.org/abs/1803.10750)
    """
    def __init__(self, opt, teacher, student, callback, checkpoint):
        super(BaseTrainer, self).__init__(opt, teacher, student, callback)

        in_size = teacher.fc.in_features  # todo: for resnet50 up it is 512 * 4 !!!!
        self.discriminator = get_discriminator(opt, in_size, checkpoint)
        log_parameter_number(self.discriminator, 'Discriminator')

        # Optimizers
        self.optimizer_G = self._define_optimizer(self.student.parameters(), opt.train.optimizer or opt.train.s_optimizer)
        self.optimizer_D = self._define_optimizer(self.discriminator.parameters(), opt.train.optimizer or opt.train.d_optimizer)
        if checkpoint is not None:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        # Loss function
        self.similarity_loss = self._define_loss(opt.train.sim_loss)
        self.adversarial_loss = Loss('BCE', opt.train.adv_loss.weight)  # todo: WGAN, ...
        self.regularizer_loss = self._define_reg(opt.train.d_reg)

        # Schedulers
        self.scheduler_G = self._define_scheduler(self.optimizer_G, opt.train.scheduler or opt.train.s_scheduler)
        self.scheduler_D = self._define_scheduler(self.optimizer_D, opt.train.scheduler or opt.train.d_scheduler)

    def _define_reg(self, opt):
        param_regs = []
        output_regs = []
        if opt is not None:
            names = opt.name if opt.name is list else list(opt.name)
            if 'L1' in names:
                param_regs.append(Regularizer('L1', opt.weight))
            if 'L2' in names:
                param_regs.append(Regularizer('L2', opt.weight))
            if 'RealAsFake' in names:
                output_regs.append(Loss('BCE', opt.weight))

            def calc_reg(outputs, targets):
                total_reg = sum(reg(self.discriminator.parameters()) for reg in param_regs) + \
                            sum(reg(outputs, targets) for reg in output_regs)
                return total_reg
            return calc_reg

        return lambda o,t: 0

    def get_snapshot(self):
        return {
            'teacher': get_model_state(self.teacher),
            'student': get_model_state(self.student),
            'discriminator': get_model_state(self.discriminator),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }

    def train(self, epoch, step):
        self.student.train()
        self.discriminator.train()
        self.teacher.eval()

        self.callback.begin_epoch(epoch, step, len(self.train_loader))
        for i, (inputs, targets) in enumerate(self.train_loader):
            step += 1

            inputs = Variable(inputs.to(self.device))

            t_outputs = self._teacher_inference(inputs)

            s_inputs = F.interpolate(inputs, scale_factor=(1.0, 0.5, 0.5), mode='nearest')
            s_outputs = self.student(s_inputs)

            d_s_outputs = self.discriminator(s_outputs['features'])
            d_t_outputs = self.discriminator(t_outputs['features'])

            # Adversarial ground truths
            valid = Variable(torch.ones(inputs.shape[0]).to(self.device), requires_grad=False)
            fake = Variable(torch.zeros(inputs.shape[0]).to(self.device), requires_grad=False)

            # -----------------
            #  Train Generator/Student
            # -----------------

            self.optimizer_G.zero_grad()
            s_adv_loss = self.adversarial_loss(d_s_outputs, valid)
            s_sim_loss = self.similarity_loss(s_outputs['logits'], t_outputs['logits'])
            s_loss = s_adv_loss + s_sim_loss

            s_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()
            d_real_loss = self.adversarial_loss(d_t_outputs, valid)
            d_fake_loss = self.adversarial_loss(d_s_outputs, fake)
            d_adv_loss = (d_real_loss + d_fake_loss) / 2
            d_reg_loss = self.regularizer_loss(d_s_outputs, valid)
            d_loss = d_adv_loss + d_reg_loss

            d_loss.backward()
            self.optimizer_D.step()

            losses = {
                'St': s_loss.mean().item(),
                'St_Adv': s_adv_loss.mean().item(),
                'St_Sim': s_sim_loss.mean().item(),
                'Ds': d_adv_loss.mean().item(),
                'Ds_Reg': d_reg_loss.mean().item()
            }
            lrs = {
                'Opt_G': self.optimizer_G.param_groups[0]['lr'],
                'Opt_D': self.optimizer_D.param_groups[0]['lr']
            }
            self.callback.iter(epoch, step, losses, lrs, inputs.shape[0])

        self.callback.end_epoch(epoch, step)
        return step

    def step_scheduler(self, epoch, metric):
        self.scheduler_G.step(epoch, metric)
        self.scheduler_D.step(epoch, metric)
