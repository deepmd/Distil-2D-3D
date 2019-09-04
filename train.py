import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

import numpy as np
import shutil
import random
import argparse
import yaml
from os import path
from libs.utils import get_logger
from models import get_models


def train(cfg, writer, logger):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Setup Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Setup Dataset

    # Setup Models
    teacher, student = get_models(cfg['models'])
    teacher = teacher.to(device)
    student = student.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, help="Configuration file to use")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = path.join("exps", path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)
