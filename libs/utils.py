"""
Misc Utility functions
"""
import os
import logging
import datetime
import torch
from collections import OrderedDict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def print_parameter_number(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_str = 'Total number of trainable parameters / all parameters in {}: {} / {}'\
        .format(model_name, trainable_params, total_params)
    print(log_str)


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def load_model(model, snapshot):
    new_state_dict = OrderedDict()
    for k, v in snapshot.items():
        head = k[:7]
        name = k[7:] if head == 'module.' else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)


def _strip_DataParallel(net):
    if isinstance(net, torch.nn.DataParallel):
        return _strip_DataParallel(net.module)
    return net


def remove_adjust_features(model):
    _strip_DataParallel(model).adjust_features = None


def get_last_features_size(model):
    return _strip_DataParallel(model).fc.in_features


def get_model_state(model):
    return _strip_DataParallel(model).state_dict()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count