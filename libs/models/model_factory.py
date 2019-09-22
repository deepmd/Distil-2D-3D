import torch

from libs.models.teachers.resnet import resnet18
from libs.models.teachers.resnet import resnet34
from libs.models.teachers.resnet import resnet50
from libs.models.teachers.resnet import resnet101
from libs.models.teachers.resnet import resnet152

from libs.models.students.c3d import C3D
from libs.models.students.r3d import R3DNet
from libs.models.students.r21d import R2Plus1DNet

from libs.models.discriminators.anc import ANCDiscriminator

from libs.utils import load_model

t_models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152}
s_models = {'c3d': C3D, 'r3d': R3DNet, 'r21d': R2Plus1DNet}
d_models = {'anc_discriminator': ANCDiscriminator}


def get_teacher_student_models(opt, checkpoint=None):
    # Create teacher model
    teacher = t_models[opt.model.t_arch](num_outputs=opt.model.n_classes) if opt.model.n_classes is not None else \
              t_models[opt.model.t_arch]()
    num_outputs = teacher.num_outputs

    # Load parameters in teacher models
    if opt.model.t_pretrain_path is not None:
        checkpoint = torch.load(opt.model.t_pretrain_path)
        teacher.load_state_dict(checkpoint)

    # Freeze teachers's weights
    for param in teacher.parameters():
        param.requires_grad = False

    # Create student model
    student = s_models[opt.model.s_arch](num_outputs=num_outputs)

    if checkpoint is None:
        checkpoint = dict()
        if opt.model.t_pretrain_path is not None:
            checkpoint['teacher'] = torch.load(opt.model.t_pretrain_path)
        if opt.model.s_pretrain_path is not None:
            checkpoint['student'] = torch.load(opt.model.s_pretrain_path)

    if 'teacher' in checkpoint:
        load_model(teacher, checkpoint['teacher'])
    if 'student' in checkpoint:
        load_model(teacher, checkpoint['student'])

    if len(opt.run.gpus) > 1:
        teacher = torch.nn.DataParallel(teacher).cuda()
        student = torch.nn.DataParallel(student).cuda()
    elif len(opt.run.gpus) > 0:
        teacher = teacher.cuda()
        student = student.cuda()

    return teacher, student


def get_discriminator(opt, input_size, checkpoint=None):
    discriminator = d_models[opt.model.d_arch](input_size=input_size)

    if checkpoint is not None and 'discriminator' in checkpoint:
        load_model(discriminator, checkpoint['discriminator'])

    if len(opt.run.gpus) > 1:
        discriminator = torch.nn.DataParallel(discriminator).cuda()
    elif len(opt.run.gpus) > 0:
        discriminator = discriminator.cuda()

    return discriminator