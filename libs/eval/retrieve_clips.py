"""Video retrieval experiment, top-k."""
from os import path, makedirs
import json

from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from libs.spatial_transforms import ToTensor
from libs.datasets.ucf101 import UCF101ClipRetrievalDataset
from libs.datasets.hmdb51 import HMDB51ClipRetrievalDataset


class RetrieveClipsEval(object):
    def __init__(self, opt, model):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.opt = opt
        self.extracted_features_path = path.join(opt.run.save_path, 'extracted_features')

        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            ToTensor(opt.eval.data.norm_value),
            transforms.Normalize(opt.eval.data.mean, opt.eval.data.std)
        ])
        if opt.eval.data.name == 'ucf101':
            train_dataset = UCF101ClipRetrievalDataset(opt.eval.data.video_path,
                                                       opt.eval.data.annotation_path,
                                                       16, 10, True, train_transforms)
        elif opt.eval.data.name == 'hmdb51':
            train_dataset = HMDB51ClipRetrievalDataset(opt.eval.data.video_path,
                                                       opt.eval.data.annotation_path,
                                                       16, 10, True, train_transforms)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=opt.eval.batch_size,
                                           shuffle=False,
                                           num_workers=opt.eval.n_threads,
                                           pin_memory=True,
                                           drop_last=True)

        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            ToTensor(opt.eval.data.norm_value),
            transforms.Normalize(opt.eval.data.mean, opt.eval.data.std)
        ])
        if opt.eval.data.name == 'ucf101':
            test_dataset = UCF101ClipRetrievalDataset(opt.eval.data.video_path,
                                                      opt.eval.data.annotation_path,
                                                      16, 10, False, test_transforms)
        elif opt.eval.data.name == 'hmdb51':
            test_dataset = HMDB51ClipRetrievalDataset(opt.eval.data.video_path,
                                                      opt.eval.data.annotation_path,
                                                      16, 10, False, test_transforms)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=opt.eval.batch_size,
                                          shuffle=False,
                                          num_workers=opt.eval.n_threads,
                                          pin_memory=True,
                                          drop_last=True)


    def _extract_feature(self):
        """Extract and save features for train split, several clips per video."""

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        with torch.no_grad():
            features = []
            classes = []
            for data in tqdm(self.train_dataloader):
                sampled_clips, idxs = data
                clips = sampled_clips.reshape((-1, 3, 16, 112, 112))
                inputs = clips.to(self.device)
                # forward
                outputs = self.model(inputs)
                # print(outputs.shape)
                # exit()
                features.append(outputs['features'].cpu().numpy().tolist())
                classes.append(idxs.cpu().numpy().tolist())
            features = np.array(features).reshape((-1, 10, outputs['features'].shape[1]))
            classes = np.array(classes).reshape((-1, 10))
            np.save(path.join(self.extracted_features_path, 'train_feature.npy'), features)
            np.save(path.join(self.extracted_features_path, 'train_class.npy'), classes)

            features = []
            classes = []
            for data in tqdm(self.test_dataloader):
                sampled_clips, idxs = data
                clips = sampled_clips.reshape((-1, 3, 16, 112, 112))
                inputs = clips.to(self.device)
                # forward
                outputs = self.model(inputs)
                features.append(outputs['features'].cpu().numpy().tolist())
                classes.append(idxs.cpu().numpy().tolist())
            features = np.array(features).reshape((-1, 10, outputs['features'].shape[1]))
            classes = np.array(classes).reshape((-1, 10))
            np.save(path.join(self.extracted_features_path, 'test_feature.npy'), features)
            np.save(path.join(self.extracted_features_path, 'test_class.npy'), classes)


    def _topk_retrieval(self):
        """Extract features from test split and search on train split features."""
        # print('Load local .npy files.')
        X_train = np.load(path.join(self.extracted_features_path, 'train_feature.npy'))
        y_train = np.load(path.join(self.extracted_features_path, 'train_class.npy'))
        X_train = np.mean(X_train,1)
        y_train = y_train[:,0]
        X_train = X_train.reshape((-1, X_train.shape[-1]))
        y_train = y_train.reshape(-1)

        X_test = np.load(path.join(self.extracted_features_path, 'test_feature.npy'))
        y_test = np.load(path.join(self.extracted_features_path, 'test_class.npy'))
        X_test = np.mean(X_test,1)
        y_test = y_test[:,0]
        X_test = X_test.reshape((-1, X_test.shape[-1]))
        y_test = y_test.reshape(-1)

        ks = [1, 5, 10, 20, 50]
        topk_correct = {k:0 for k in ks}

        distances = cosine_distances(X_test, X_train)
        indices = np.argsort(distances)

        for k in ks:
            # print(k)
            top_k_indices = indices[:, :k]
            # print(top_k_indices.shape, y_test.shape)
            for ind, test_label in zip(top_k_indices, y_test):
                labels = y_train[ind]
                if test_label in labels:
                    # print(test_label, labels)
                    topk_correct[k] += 1

        for k in ks:
            correct = topk_correct[k]
            total = len(X_test)
            print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

        with open(path.join(self.extracted_features_path, 'topk_correct.json'), 'w') as fp:
            json.dump(topk_correct, fp)

        ret_acc = {'Top-{}'.format(k): topk_correct[k]/len(X_test) for k in ks}
        return ret_acc['Top-5'], ret_acc

    def eval(self):
        if not path.exists(self.extracted_features_path):
            makedirs(self.extracted_features_path)
        self._extract_feature()
        return self._topk_retrieval()

