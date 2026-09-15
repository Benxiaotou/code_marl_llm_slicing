
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

from scipy.fftpack import fft
import math


class ObtainTask(object):

    def __init__(self, character_folders, class_num=8, train_num=70, test_num=30):

        self.character_folders = character_folders
        self.class_num = class_num
        self.train_num = train_num
        self.test_num = test_num

        self.train_files = []
        self.test_files = []
        self.train_labels = []
        self.test_labels = []

        class_folders = random.sample(self.character_folders, self.class_num)
        index = 0
        for class_folder in class_folders:
            (file, label) = class_folder
            assert len(file) >= self.train_num + self.test_num,\
                "The sum of the samples used for training and testing exceeds the total number of samples for this class"
            np.random.shuffle(file)
            self.train_files += list(file[:self.train_num, :])
            self.test_files += list(file[self.train_num:self.train_num + self.test_num, :])
            self.train_labels += [index for i in range(self.train_num)]
            self.test_labels += [index for i in range(self.test_num)]
            index += 1


def add_noise(x, snr, seed=None):#高斯白噪声
    '''
    参考：https://blog.csdn.net/weixin_43545253/article/details/113721360
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    if seed is not None:
        np.random.seed(seed=seed)

    noise = np.random.randn(len(x))
    npower = np.sum(noise ** 2) / len(noise)
    xpower = np.sum(x ** 2) / len(x)
    n_variance = xpower / (10 ** (snr / 10.0))
    noise = noise * np.sqrt(n_variance / npower)
    return x + noise

def SNR_singlech(x_o,x_n):
    # x_o :original signal
    # x_n:noisy signal(ie. original signal + noise signal)
    x_o_power = np.sum(x_o ** 2) / len(x_o)
    x_n_power = np.sum((x_o-x_n) ** 2) / len((x_o-x_n))
    snr = 10 * math.log10(x_o_power/x_n_power)
    return snr

def get_TL_dataloader(character_folders, train_batch_size=100,
                      class_num=8, train_num=200, test_num=100,
                      shuffle=True, snr='NO', model_type='No'):    #train_num和test_num都是每一类的数目

    task = ObtainTask(character_folders, class_num=class_num, train_num=train_num, test_num=test_num)

    train_dataset = ObtainDataset(task, split='train', snr=snr, model_type=model_type)#
    test_dataset = ObtainDataset(task, split='test', snr=snr, model_type=model_type)#

    train_loader= DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    return train_loader, test_loader

class ObtainDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None, snr='No',model_type='No'):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.snr = snr
        self.sample_files = self.task.train_files if self.split == 'train' else self.task.test_files
        self.labels = np.array(self.task.train_labels if self.split == 'train' else self.task.test_labels).reshape(-1)
        self.model_type = model_type
        assert self.model_type != 'No', "The MODEL (model_type) should not be 'No."

    def __getitem__(self, idx):
        sample = self.sample_files[idx]
        if type(self.snr) != str:
            sample = add_noise(x=sample, snr=self.snr)

        if self.model_type == 'TCN':
            sample = abs(fft(sample - np.mean(sample)))[0:1024].reshape([1, 1024])
        else:
            print("The MODEL (model_type) is error.")
        # if self.dt == 'fft':
        #     sample = abs(fft(sample - np.mean(sample)))[0:1024].reshape([1, 1024])
        #
        # if self.mt == '1d' and self.dt == 't':
        #     sample = sample[0:1024].reshape([1, 1024])
        # elif self.mt == '2d':
        #     sample = sample[0:1024].reshape([32, 32])
        return sample, np.int64(self.labels[idx])

    def __len__(self):
        return len(self.sample_files)




