import argparse
import yaml
from os import path
import random


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="2D to 3D Knowledge Distillation")
        self.parser.add_argument(
            "--config",
            type=str,
            help="Configuration file to use",
        )

        with open('configs/dataset_config.yml') as fp:
            self.dataset_cfg = yaml.load(fp)

        with open('configs/default_config.yml') as fp:
            self.default_opt = yaml.load(fp)

        self.default_opt['train']['s_optimizer'] = self.default_opt['train']['optimizer']
        self.default_opt['train']['d_optimizer'] = self.default_opt['train']['optimizer']
        self.default_opt['train']['s_scheduler'] = self.default_opt['train']['scheduler']
        self.default_opt['train']['d_scheduler'] = self.default_opt['train']['scheduler']

    def _update_params(self, template, source):
        result = DictAsMember()
        for key, value in template.items():
            if key in source and source[key] is not None:
                result[key] = source[key]
            elif not isinstance(value, dict):
                # assume the template value is a default
                result[key] = value
            else:
                # recurse
                result[key] = self.update_nested_params(value, source)
        return result

    def parse(self):
        args = self.parser.parse_args()
        with open(args.config) as fp:
            override_opt = yaml.load(fp)

        opt = self._update_params(self.default_opt, override_opt)

        if opt['run']['gpus'] == -1:
            opt['run']['gpus'] = []
        elif opt['run']['gpus'] is not list:
            opt['run']['gpus'] = list(opt['run']['gpus'])

        if 'train' not in override_opt:
            del opt['train']
            opt['no_train'] = True
        else:
            opt['train']['batch_size'] *= max(1, len(opt['run']['gpus']))

            train_dataset_cfg = self.dataset_cfg[opt['train']['data']['name']]
            opt['train']['data']['video_path'] = path.join(train_dataset_cfg['root_path'],
                                                           train_dataset_cfg['jpg_video_path'])
            opt['train']['data']['annotation_path'] = path.join(train_dataset_cfg['root_path'],
                                                                train_dataset_cfg['annotation_path'])

            if opt['train']['data']['mean'] is None:
                opt['train']['data']['mean'] = [v / opt['train']['data']['norm_value'] for v in train_dataset_cfg['mean']]
            if opt['train']['data']['std'] is None:
                opt['train']['data']['std'] = [v / opt['train']['data']['norm_value'] for v in train_dataset_cfg['std']]

            opt['train']['data']['scales'] = [opt['train']['data']['initial_scale']]
            for i in range(1, opt['train']['data']['n_scales']):
                opt['train']['data']['scales'].append(opt['train']['data']['scales'][-1] * opt['train']['data']['scale_step'])

            assert any(k in override_opt['train'] for k in ('s_optimizer', 'd_optimizer')) != \
                   'optimizer' in override_opt['train']
            if 's_optimizer' not in override_opt['train']:
                del opt['train']['s_optimizer']
            if 'd_optimizer' not in override_opt['train']:
                del opt['train']['d_optimizer']

            assert any(k in override_opt['train'] for k in ('s_scheduler', 'd_scheduler')) != \
                   'scheduler' in override_opt['train']
            if 's_scheduler' not in override_opt['train']:
                del opt['train']['s_scheduler']
            if 'd_scheduler' not in override_opt['train']:
                del opt['train']['d_scheduler']

            if 'adv_loss' not in override_opt['train']:
                opt['train']['adv_loss'] = None
            if 'd_reg' not in override_opt['train']:
                opt['train']['d_reg'] = None

        if 'eval' not in override_opt:
            del opt['eval']
            opt['no_eval'] = True
        else:
            opt['eval']['batch_size'] *= max(1, len(opt['run']['gpus']))

            eval_dataset_cfg = self.dataset_cfg[opt['eval']['data']['name']]
            opt['eval']['data']['video_path'] = path.join(eval_dataset_cfg['root_path'],
                                                          eval_dataset_cfg['avi_video_path'])
            opt['eval']['data']['annotation_path'] = path.join(eval_dataset_cfg['root_path'],
                                                               eval_dataset_cfg['annotation_path'])

            if opt['eval']['data']['mean'] is None:
                opt['eval']['data']['mean'] = [v / opt['eval']['data']['norm_value'] for v in eval_dataset_cfg['mean']]
            if opt['eval']['data']['std'] is None:
                opt['eval']['data']['std'] = [v / opt['eval']['data']['norm_value'] for v in eval_dataset_cfg['std']]

        if opt['run']['exp_id'] is None:
            opt['run']['exp_id'] = random.randint(1, 100000)
        opt['run']['save_path'] = path.join(opt['run']['exp_path'], opt['run']['exp_id'])

        opt['cfg_file'] = args.config

        return opt
