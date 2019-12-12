import torch
from torch.nn import functional as F

from libs.train.base_trainer import BaseTrainer
from libs.train.loss import Loss, Regularizer
from libs.models.model_factory import get_discriminator
from libs.utils import get_model_state, get_last_features_size, print_parameter_number


class ANCTrainer(BaseTrainer):
    """ Belagiannis et al., Adversarial Network Compression
        (http://arxiv.org/abs/1803.10750)
    """
    def __init__(self, opt, teacher, student, callback, checkpoint):
        super(ANCTrainer, self).__init__(opt, teacher, student, callback)

        # Create discriminator model
        in_size = get_last_features_size(teacher)
        self.discriminator = get_discriminator(opt, in_size, checkpoint)
        print_parameter_number(self.discriminator, 'Discriminator')

        # Define two optimizers for generator/student and discriminator
        self.optimizer_G = self._define_optimizer(self.student.parameters(), opt.train.optimizer or opt.train.s_optimizer)
        self.optimizer_D = self._define_optimizer(self.discriminator.parameters(), opt.train.optimizer or opt.train.d_optimizer)
        # Load optimizer states from checkpoint if available
        if checkpoint is not None:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        # Define loss functions
        self.similarity_loss = Loss(opt.train.sim_loss.name, opt.train.sim_loss.weight)
        self.adversarial_loss = Loss('BCE', opt.train.adv_loss.weight, logit_target=False)  # todo: WGAN, ...
        self.regularizer_loss = self._define_reg(opt.train.d_reg)

        # Define two schedulers for generator/student and discriminator
        self.scheduler_G = self._define_scheduler(self.optimizer_G, opt.train.scheduler or opt.train.s_scheduler)
        self.scheduler_D = self._define_scheduler(self.optimizer_D, opt.train.scheduler or opt.train.d_scheduler)

    def _define_reg(self, opt):
        param_regs = []
        output_regs = []
        if opt is not None:
            names = opt.name if isinstance(opt.name, list) else [opt.name]
            # Add discriminator-parameter and discriminator-output regularizers based on specified options
            if 'L1' in names:
                param_regs.append(Regularizer('L1', opt.weight))
            if 'L2' in names:
                param_regs.append(Regularizer('L2', opt.weight))
            if 'FakeAsReal' in names:
                output_regs.append(Loss('BCE', opt.weight, logit_target=False))

            def calc_reg(outputs, targets):
                total_reg = -sum(reg(self.discriminator.parameters()) for reg in param_regs) + \
                            sum(reg(outputs, targets) for reg in output_regs)
                return total_reg
            # Return calc_rec function which calculates sum of all regularizers
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

        # Train loop for an epoch
        for inputs, targets in self.train_loader:
            step += 1
            batch_size = inputs.shape[0]

            inputs = inputs.to(self.device)

            # Run teacher model and perform temporal pooling on results
            t_outputs = self._teacher_inference(inputs)

            # Resize inputs for student model
            s_inputs = F.interpolate(inputs, scale_factor=(1.0, 0.5, 0.5), mode='bilinear')
            # Run student model with extra layer(s) to adjust its features size to the same size as teacher
            s_outputs = self.student(s_inputs, adjust_features=True)
            # Run student model with dropout to be used as "adversarial samples"
            s_outputs_adv = self.student(s_inputs, dropout=0.5, adjust_features=True)

            # Create adversarial ground truths
            valid = torch.ones(batch_size).to(self.device)
            fake = torch.zeros(batch_size).to(self.device)

            # -----------------
            #  Train Generator/Student
            # -----------------

            d_s_outputs_adv = self.discriminator(s_outputs_adv['features']).squeeze(dim=1)
            s_adv_loss = self.adversarial_loss(d_s_outputs_adv, valid)
            s_sim_loss = self.similarity_loss(s_outputs['logits'], t_outputs['logits'])
            s_loss = s_adv_loss + s_sim_loss

            self.optimizer_G.zero_grad()
            s_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # s_outputs, and s_outputs_adv should be detached otherwise loss.backward() will
            # be calculated on the student network in addition to the discriminator network
            d_t_outputs = self.discriminator(t_outputs['features']).squeeze(dim=1)
            d_s_outputs = self.discriminator(s_outputs['features'].detach()).squeeze(dim=1)
            d_s_outputs_adv = self.discriminator(s_outputs_adv['features'].detach()).squeeze(dim=1)
            d_real_loss = self.adversarial_loss(d_t_outputs, valid)
            d_fake_loss = self.adversarial_loss(d_s_outputs, fake)
            d_adv_loss = d_real_loss + d_fake_loss
            d_reg_loss = self.regularizer_loss(d_s_outputs_adv, valid)
            d_loss = d_adv_loss + d_reg_loss

            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            losses = {
                'St': s_loss.item(),
                'St_Adv': s_adv_loss.item(),
                'St_Sim': s_sim_loss.item(),
                'Ds': d_adv_loss.item(),
                'Ds_Reg': d_reg_loss.item()
            }
            lrs = {
                'Opt_G': self.optimizer_G.param_groups[0]['lr'],
                'Opt_D': self.optimizer_D.param_groups[0]['lr']
            }
            self.callback.end_iter(epoch, step, losses, lrs, batch_size)

        self.callback.end_epoch(epoch, step)
        return step

    def step_scheduler(self, epoch, metric):
        self.scheduler_G(epoch, metric)
        self.scheduler_D(epoch, metric)
