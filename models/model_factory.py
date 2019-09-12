import torch

from .teachers.resnet import resnet18
from .teachers.resnet import resnet34
from .teachers.resnet import resnet50
from .teachers.resnet import resnet101
from .teachers.resnet import resnet152

from .students.c3d import C3D
from .students.r3d import R3DNet
from .students.r21d import R2Plus1DNet

t_models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152}
s_models = {'c3d': C3D, 'r3d': R3DNet, 'r21d': R2Plus1DNet}


def get_models(t_arch, s_arch, num_outputs, t_pretrain_path=None, freeze_t_bbon=True):
    # Create teacher model
    teacher = t_models[t_arch](num_outputs=num_outputs)

    # Load parameters in teacher models
    if t_pretrain_path is not None:
        checkpoint = torch.load(t_pretrain_path)
        teacher.load_state_dict(checkpoint)

    # Freeze teachers's weights
    if freeze_t_bbon:
        for param in teacher.parameters():
            param.requires_grad = False

    # Create student model
    student = s_models[s_arch](num_outputs=num_outputs)

    return teacher, student

    # # for t in args.teachers:
    # #     if t in model_map:
    # #         net = model_map[t](args)
    # #         net.__name__ = t
    # #         teachers.append(net)
    # #
    # # assert len(teachers) > 0, "teachers must be in %s" % " ".join(model_map.keys)
    #
    # # Initialize student model
    # # assert args.student in model_map, "students must be in %s" % " ".join(model_map.keys)
    # # student = model_map[args.student](args)
    #
    # # Model setup
    # # if device == "cuda":
    # #     cudnn.benchmark = True
    #
    # for i, teacher in enumerate(teachers):
    #     for p in teacher.parameters():
    #         p.requires_grad = False
    #     teacher = teacher.to(device)
    #     if device == "cuda":
    #         teachers[i] = torch.nn.DataParallel(teacher)
    #         teachers[i].__name__ = teacher.__name__

    # Load parameters in teacher models
    # for teacher in teachers:
    #     if teacher.__name__ != "shake_shake":
    #         checkpoint = torch.load('./checkpoint/%s/ckpt.t7' % teacher.__name__)
    #         model_dict = teacher.state_dict()
    #         pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
    #         model_dict.update(pretrained_dict)
    #         teacher.load_state_dict(model_dict)
    #         print("teacher %s acc: ", (teacher.__name__, checkpoint['acc']))
    #
    # student = student.to(device)



    # if device == "cuda":
    #     out_dims = student.out_dims
    #     student = torch.nn.DataParallel(student)
    #     student.out_dims = out_dims
    #
    # if args.teacher_eval:
    #     for teacher in teachers:
    #         teacher.eval()


