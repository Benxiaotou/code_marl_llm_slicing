
import torch
import random
import os
import numpy as np
from scipy.io import loadmat
import logging
import math


def get_class(sample):
    return sample.split('\\')[-2]

def UPBFolders(CLASS_NUM):
    root = "C:\\Users\\Administrator\\Desktop\\code\\dataset\\UPB_bearing_dataset"
    labels = ['KA01', 'KA03', 'KA05', 'KA07', 'KA08', 'KI01', 'KI03', 'KI07'] \
             + ['K001'] + ['KA04','KA16', 'KB23', 'KB27', 'KI18', 'KI21']

    folds = [os.path.join(root, label) for label in labels]

    samples = dict()
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []
    for c in folds[:-CLASS_NUM]:

        name0 = 'N15_M07_F10_'
        name1 = c.split('\\')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)) for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])

        train_labels.append(get_class(temp))
        train_files.append(file)

    for c in folds[-CLASS_NUM:]:

        name1 = c.split('\\')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)) for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])


        test_labels.append(get_class(temp))
        test_files.append(file)

    metatrain_folders = list(zip(train_files, train_labels))
    metatest_folders = list(zip(test_files, test_labels))
    return metatrain_folders, metatest_folders

def SEUFolders(num_samples_per_class=300):
    root0 = 'C:\\Users\\Administrator\\Desktop\\code\\dataset\\SEU_gearbox_dataset'
    root1 = '\\processed_data'

    labels = ['ball', 'inner', 'outer', 'comb', 'Chipped', 'Miss', 'Root', 'Surface', 'Health']

    labels_dict = dict()
    labels_dict['ball'] = ['DE_ball_20_0_x', 'DE_ball_30_2_x']
    labels_dict['inner'] = ['DE_inner_20_0_x', 'DE_inner_30_2_x']
    labels_dict['outer'] = ['DE_outer_20_0_x', 'DE_outer_30_2_x']
    labels_dict['comb'] = ['DE_comb_20_0_x', 'DE_comb_30_2_x']
    labels_dict['Chipped'] = ['DE_Chipped_20_0', 'DE_Chipped_30_2']
    labels_dict['Miss'] = ['DE_Miss_20_0', 'DE_Miss_30_2']
    labels_dict['Root'] = ['DE_Root_20_0', 'DE_Root_30_2']
    labels_dict['Surface'] = ['DE_Surface_20_0', 'DE_Surface_30_2']
    labels_dict['Health'] = ['DE_Health_20_0', 'DE_Health_30_2']

    NUM_POINTS = 2048
    NUM_SAMPLES = math.ceil(num_samples_per_class/len(labels_dict['ball']))
    logging.info("The dataset type is SEU." + " The num_points is " + str(NUM_POINTS) +
                 ". The num_samples is " + str(NUM_SAMPLES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = dict()
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []
    for c in labels:

        name0 = root0 + root1
        name1 = labels_dict[c]
        temps = [os.path.join(name0, str(x)) for x in name1]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('\\')[-1]].reshape(-1)
            data1 = sample_split(data0, num_points=NUM_POINTS, num_samples=NUM_SAMPLES)

            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])
            print(i)
        print(c)

        train_labels.append(c)
        train_files.append(file)

    train_folders = list(zip(train_files, train_labels))
    return train_folders

def CWRUFolders(num_samples_per_class=2400):
    root0 = 'C:\\Users\\Administrator\\Desktop\\code\\dataset\\CWRU_bearing_dataset'
    root1 = '\\12k Drive End Bearing Fault Data'
    root2 = '\\Normal Baseline Data'
    labels = ['BF7', 'BF14', 'BF21', 'IF7', 'IF14', 'IF21', 'OF7', 'OF14', 'OF21', 'NC']

    labels_dict = dict()
    labels_dict['BF7'] = [118, 119, 120, 121]
    labels_dict['BF14'] = [185, 186, 187, 188]
    labels_dict['BF21'] = [222, 223, 224, 225]
    labels_dict['IF7'] = [105, 106, 107, 108]
    labels_dict['IF14'] = [169, 170, 171, 172]
    labels_dict['IF21'] = [209, 210, 211, 212]
    labels_dict['OF7'] = [130, 131, 132, 133]
    labels_dict['OF14'] = [197, 198, 199, 200]
    labels_dict['OF21'] = [234, 235, 236, 237]
    labels_dict['NC'] = [97, 98, 99, 100]
    NUM_POINTS = 2048
    NUM_SAMPLES = math.ceil(num_samples_per_class/len(labels_dict['BF7']))
    logging.info("The dataset type is CWRU." + " The num_points is " + str(NUM_POINTS) +
                 ". The num_samples is " + str(NUM_SAMPLES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = dict()
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []
    for c in labels:

        name0 = root0 + root1 if c!='NC' else root0 + root2
        name1 = labels_dict[c]
        name2 = '_DE_time'
        temps = [os.path.join(name0, str(x)) for x in name1]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)['X' + temp.split('\\')[-1] + name2].reshape(-1) if int(temp.split('\\')[-1])>=100 \
                else loadmat(temp)['X0' + temp.split('\\')[-1] + name2].reshape(-1)
            data1 = sample_split(data0, num_points=NUM_POINTS, num_samples=NUM_SAMPLES)

            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])
            print(i)
        print(c)

        train_labels.append(c)
        train_files.append(file)

    train_folders = list(zip(train_files, train_labels))
    return train_folders

def sample_split(data0, num_points=2048, num_samples=800):
    if num_points*num_samples <= data0.size:
        data1 = data0[:data0.size // num_points * num_points].reshape(-1, num_points)
        data1 = data1[:num_samples]
    else:
        x_tap = (data0.size-num_points) // (num_samples-1)
        data1=[]
        for i in range(num_samples):
            if i == 0:
                data1 = data0[i*x_tap : i*x_tap+num_points]
            else:
                data1 = np.vstack([data1, data0[i*x_tap : i*x_tap+num_points]])
    return data1
