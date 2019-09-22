import torch
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from progress.bar import Bar
import numpy as np
import shutil
import random
import os
import time
import datetime
from collections import defaultdict

from libs.utils import get_logger, log_parameter_number, AverageMeter
from libs.models.model_factory import get_teacher_student_models
from libs.train.trainer_factory import get_trainer
from libs.eval.evaluator_factory import get_evaluator
from libs.opts import opts


class TrainCallback(object):
    def __init__(self, exp_id, logging_n, writer, logger):
        self.writer = writer
        self.logger = logger
        self.exp_id = exp_id
        self.logging_n = logging_n

    def begin_epoch(self, epoch, step, total_iters):
        self.bar = Bar('{}'.format(self.exp_id), max=total_iters)
        self.begin_step = step
        self.total_iters = total_iters
        self.bar_avg_loss = defaultdict(lambda: AverageMeter())
        self.log_avg_loss = defaultdict(lambda: AverageMeter())
        self.bar_batch_time = AverageMeter()
        self.log_batch_time = AverageMeter()
        self.start = time.time()
        self.last_lrs = dict()

    def enb_epoch(self, epoch, step):
        self.bar.finish()
        # ============ TensorBoard logging ============#
        info = {('param/LR_' + l): v for l, v in self.last_lrs.items}
        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)


    def iter(self, epoch, step, losses, lrs, batch_size):
        # ============ Progressbar & logging ============#
        iter_id = step - self.begin_step
        self.bar_batch_time.update(time.time() - self.start)
        self.log_batch_time.update(time.time() - self.start)

        Bar.suffix = '[{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} |Batch: {batch:.1f} '.format(
            epoch, iter_id, self.total_iters, total=self.bar.elapsed_td, eta=self.bar.eta_td, batch=self.bar_batch_time.avg)
        if len(losses) == 1:
            self.bar_avg_loss['LOSS'].update(list(losses)[0], batch_size)
            self.log_avg_loss['LOSS'].update(list(losses)[0], batch_size)
            Bar.suffix += '|Loss: {:.4f} '.format(self.bar_avg_loss['LOSS'].avg)
        else:
            Bar.suffix += '|Loss '
            for l_name, l_val in losses.items():
                self.bar_avg_loss[l_name].update(l_val, batch_size)
                self.log_avg_loss[l_name].update(l_val, batch_size)
                Bar.suffix += '{}: {:.4f} '.format(l_name, self.bar_avg_loss[l_name].avg)
        if len(lrs) == 1:
            Bar.suffix += '|LR: {:.4f} '.format(list(lrs)[0])
        else:
            Bar.suffix += '|LR '
            for lr_name, lr_val in lrs.items():
                Bar.suffix += '|{}: {:.6f} '.format(lr_name, lr_val)
        self.last_lrs = lrs
        self.bar.next()

        if step % self.logging_n == 0:
            if len(losses) == 1:
                log_str = 'Train [{0}, {1}][{2}/{3}]|Tot: {total:} |Batch: {batch:.1f}'.format(
                    epoch, step, iter_id, self.total_iters, total=self.bar.elapsed_td, batch=self.log_batch_time.avg)
                if len(losses) == 1:
                    log_str += '|Loss: {:.4f} '.format(self.log_avg_loss['LOSS'].avg)
                else:
                    log_str += '|Loss '
                    for l_name in losses.keys():
                        log_str += '{}: {:.4f} '.format(l_name, self.log_avg_loss[l_name].avg)
                if len(lrs) == 1:
                    log_str += '|LR: {:.4f} '.format(list(lrs)[0])
                else:
                    log_str += '|LR '
                    for lr_name, lr_val in lrs.items():
                        log_str += '|{}: {:.6f} '.format(lr_name, lr_val)
                self.logger.info(log_str)
                for log_avg in self.log_avg_loss.values():
                    log_avg.reset()
                self.log_batch_time.reset()

        # ============ TensorBoard logging ============#
        info = {('loss/'+l): v for l,v in losses.items}
        for tag, value in info.items():
            writer.add_scalar(tag, value, step)

        self.start = time.time()


def save_checkpoint(path, trainer, epoch, step, metric, best_metric):
    checkpoint = trainer.get_snapshot()
    checkpoint['epoch'] = epoch
    checkpoint['step'] = step
    checkpoint['metric'] = metric
    checkpoint['best_metric'] = best_metric
    torch.save(checkpoint, path)


def evaluate(evaluator, epoch=0, writer=None, logger=None):
    start = time.time()
    print('Starting Evaluation ...')
    metric = evaluator.eval()
    log_str = 'Evaluation Result: {:.3f} (t={:.0f}s)'.format(metric, time.time() - start)
    print(log_str)
    if logger is not None:
        logger.info(log_str)
    if writer is not None:
        writer.add_scalar('metric/retrieval', metric, epoch)
    return metric


def train(opt, writer, logger):
    # Load checkpoint
    checkpoint = None
    begin_epoch = 0
    step = 0
    best_metric = 0
    metric = 0
    if not opt.no_train and opt.train.resume_path is not None:
        print('loading checkpoint {}'.format(opt.train.resume_path))
        checkpoint = torch.load(opt.train.resume_path)
        begin_epoch = checkpoint['epoch']
        step = checkpoint['step']
        metric = checkpoint['metric']
        best_metric = checkpoint['best_metric']

    # Setup Models
    teacher, student = get_teacher_student_models(opt, checkpoint)

    # Print number of parameters
    log_parameter_number(teacher, 'Teacher')
    log_parameter_number(student, 'Student')

    if not opt.no_eval:
        evaluator = get_evaluator(opt, student)

    if not opt.no_train:
        train_callback = TrainCallback(opt.run.exp_id, opt.train.logging_n, writer, logger)
        trainer = get_trainer(opt, teacher, student, train_callback, checkpoint)
        for epoch in range(begin_epoch+1, opt.train.n_epochs+1):
            step = trainer.train(epoch, step)
            if not opt.no_eval and epoch % opt.eval.epoch_n == 0:
                metric = evaluate(evaluator, epoch, writer, logger)
                if metric >= best_metric:
                    best_metric = metric
                    best_save_path = os.path.join(opt.result_path, 'best.pth'.format(epoch))
                    save_checkpoint(best_save_path, trainer, epoch, step, metric, best_metric)
            if epoch % opt.train.checkpoint == 0:
                chk_save_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
                save_checkpoint(chk_save_path, trainer, epoch, step, metric, best_metric)
            trainer.step_scheduler(epoch, metric)

    elif not opt.no_eval:
        evaluate(evaluator)


if __name__ == '__main__':
    opt = opts().parse()

    print("RUNDIR: {}".format(opt.run.save_path))
    shutil.copy(opt.config_file, opt.run.save_path)

    # Setup logger
    writer = SummaryWriter(log_dir=opt.run.save_path)
    logger = get_logger(opt.run.save_path)
    log_str = "Starting Experiment {} at {:%Y-%m-%d %H:%M:%S}".format(opt.run.exp_id, datetime.date.today())
    print(log_str)
    logger.info(log_str)

    # Setup seeds
    torch.manual_seed(opt.run.manual_seed)
    torch.cuda.manual_seed(opt.run.manual_seed)
    np.random.seed(opt.run.manual_seed)
    random.seed(opt.run.manual_seed)

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in opt.run.gpus])
    if len(opt.run.gpus) > 0:
        cudnn.benchmark = True

    train(opt, writer, logger)
