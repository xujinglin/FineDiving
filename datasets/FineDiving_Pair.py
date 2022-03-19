import torch
import numpy as np
import os
import pickle
import random
import glob
from os.path import join
from PIL import Image

class FineDiving_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform
        self.random_choosing = args.random_choosing
        self.action_number_choosing = args.action_number_choosing
        self.length = args.frame_length
        self.voter_number = args.voter_number

        # file path
        self.data_root = args.data_root
        self.data_anno = self.read_pickle(args.label_path)
        with open(args.train_split, 'rb') as f:
            self.train_dataset_list = pickle.load(f)
        with open(args.test_split, 'rb') as f:
            self.test_dataset_list = pickle.load(f)

        self.action_number_dict = {}
        self.difficulties_dict = {}
        if self.subset == 'train':
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.action_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.action_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def preprocess(self):
        for item in self.train_dataset_list:
            dive_number = self.data_anno.get(item)[0]
            if self.action_number_dict.get(dive_number) is None:
                self.action_number_dict[dive_number] = []
            self.action_number_dict[dive_number].append(item)
        if self.subset == 'test':
            for item in self.test_dataset_list:
                dive_number = self.data_anno.get(item)[0]
                if self.action_number_dict_test.get(dive_number) is None:
                    self.action_number_dict_test[dive_number] = []
                self.action_number_dict_test[dive_number].append(item)

    def check_exemplar_dict(self):
        if self.subset == 'train':
            for key in sorted(list(self.action_number_dict.keys())):
                file_list = self.action_number_dict[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key
        if self.subset == 'test':
            for key in sorted(list(self.action_number_dict_test.keys())):
                file_list = self.action_number_dict_test[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key

    def load_video(self, video_file_name):
        image_list = sorted((glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))

        if len(image_list) >= self.length:
            start_frame = int(image_list[0].split("/")[-1][:-4])
            end_frame = int(image_list[-1].split("/")[-1][:-4])
            frame_list = np.linspace(start_frame, end_frame, self.length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

            video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
            frames_labels = [self.data_anno.get(video_file_name)[4][i] for i in image_frame_idx]
            frames_catogeries = list(set(frames_labels))
            frames_catogeries.sort(key=frames_labels.index)
            transitions = [frames_labels.index(c) for c in frames_catogeries]
            return self.transforms(video), np.array([transitions[1]-1,transitions[-1]-1]), np.array(frames_labels)
        else:
            video = [Image.open(image_list[i]) for i in range(len(image_list))]
            video_1 = self.transforms(video)
            C, T, H, W = video_1.size()
            video_1 = video_1.unsqueeze(-1).expand(C, T, H, W, 2)
            video_1 = video_1.reshape(C, -1, H, W)
            C1, T1, H1, W1 = video_1.size()

            frame_labels_1 = self.data_anno.get(video_file_name)[4]
            TT = frame_labels_1.shape[0]
            frame_labels_1 = torch.from_numpy(frame_labels_1).unsqueeze(-1).expand(T, 2)
            frame_labels_1 = frame_labels_1.reshape(-1)
            TT1 = frame_labels_1.shape[0]
            frame_labels_1 = frame_labels_1.tolist()

            assert T1 == TT1 and T == TT
            select_space = np.linspace(0, TT1 - 1, self.length).astype(np.int)
            select_frame_list = [select_space[i] for i in range(self.length)]

            video_1 = torch.cat([video_1[:, ii, :, :].unsqueeze(1) for ii in select_frame_list], 1)
            frame_labels_1 = [frame_labels_1[ii] for ii in select_frame_list]
            frames_catogeries = list(set(frame_labels_1))
            frames_catogeries.sort(key=frame_labels_1.index)
            transitions = [frame_labels_1.index(c) for c in frames_catogeries]
            return video_1, np.array([transitions[1]-1,transitions[-1]-1]), np.array(frame_labels_1)


    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample_1  = self.dataset[index]
        data = {}
        data['video'], data['transits'], data['frame_labels'] = self.load_video(sample_1)
        data['number'] = self.data_anno.get(sample_1)[0]
        data['final_score'] = self.data_anno.get(sample_1)[1]
        data['difficulty'] = self.data_anno.get(sample_1)[2]
        data['completeness'] = (data['final_score'] / data['difficulty'])

        # choose a exemplar
        if self.subset == 'train':
            # train phrase
            if self.action_number_choosing == True:
                file_list = self.action_number_dict[self.data_anno[sample_1][0]].copy()
            elif self.DD_choosing == True:
                file_list = self.difficulties_dict[self.data_anno[sample_1][2]].copy()
            else:
                # randomly
                file_list = self.train_dataset_list.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            target['video'], target['transits'], target['frame_labels'] = self.load_video(sample_2)
            target['number'] = self.data_anno.get(sample_2)[0]
            target['final_score'] = self.data_anno.get(sample_2)[1]
            target['difficulty'] = self.data_anno.get(sample_2)[2]
            target['completeness'] = (target['final_score'] / target['difficulty'])
            return data, target
        else:
            # test phrase
            if self.action_number_choosing:
                train_file_list = self.action_number_dict[self.data_anno[sample_1][0]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            elif self.DD_choosing:
                train_file_list = self.difficulties_dict[self.data_anno[sample_1][2]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'], tmp['transits'], tmp['frame_labels'] = self.load_video(item)
                tmp['number'] = self.data_anno.get(item)[0]
                tmp['final_score'] = self.data_anno.get(item)[1]
                tmp['difficulty'] = self.data_anno.get(item)[2]
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])
                target_list.append(tmp)
            return data, target_list

    def __len__(self):
        return len(self.dataset)
