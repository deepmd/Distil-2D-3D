from libs.train.anc import ANCTrainer
from libs.train.simple import SimpleTrainer


def get_trainer(opt, teacher, student, callback=None, checkpoint=None):
    if opt.train.method == 'ANC':
        return ANCTrainer(opt, teacher, student, callback, checkpoint)
    elif opt.train.method == 'Simple':
        return SimpleTrainer(opt, teacher, student, callback, checkpoint)
    else:
        raise ValueError("There's no trainer method named '{}'!".format(opt.train.method))
