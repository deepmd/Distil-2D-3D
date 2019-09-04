from .ucf101 import UCF101Dataset
from .hmdb51 import HMDB51Dataset
from .kinetics import Kinetics


def get_dataset(name):
    """get_loader

    :param name: name of the dataset
    """
    return {
        'kinetics': Kinetics,
        'ucf101': UCF101Dataset,
        'hmdb51': HMDB51Dataset
    }[name]