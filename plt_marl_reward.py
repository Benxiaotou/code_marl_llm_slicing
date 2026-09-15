import os
import shutil
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import utlis.finetune_generator as fg
import argparse
import pickle
import logging
import collections
import heapq
import random
import matplotlib.pyplot as plt
import pandas as pd

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
# from stable_baselines3 import results_plotter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from PIL import Image
from collections import deque
import numpy as np
# from utlis.general_function import setlogger, setup_seed


def _maxmin_Normalization(x, max, min):
    x = (x - min) / (max - min)
    return x

if __name__ == "__main__":
    BATCH_SIZE = 16
    NSTEPS = 1000000
    GAMMA = 0.90
    REG = 0.01
    SEED = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------------path---------------------------
    root = "C:\\Users\\Lenovo\\Desktop\\paper_code\\pxl_4_code\\code_marl_v0\\result"

    path_model_train_log = os.path.join(root, 'model_train_log')
    if not os.path.exists("%s" % path_model_train_log):
        os.makedirs("%s" % path_model_train_log)

    path_ppo_model_save = os.path.join(root, 'ppo_model_save')
    if not os.path.exists("%s" % path_ppo_model_save):
        os.makedirs("%s" % path_ppo_model_save)

    path_reward_save = os.path.join(root, 'reward_save')
    if not os.path.exists("%s" % path_reward_save):
        os.makedirs("%s" % path_reward_save)

    path_Agents_save = os.path.join(root, 'Agents_save')
    if not os.path.exists("%s" % path_Agents_save):
        os.makedirs("%s" % path_Agents_save)

    path_F_save = os.path.join(root, 'F_save')
    if not os.path.exists("%s" % path_F_save):
        os.makedirs("%s" % path_F_save)

    path_logger = os.path.join(root, 'logger_save')
    if not os.path.exists("%s" % path_logger):
        os.makedirs("%s" % path_logger)


    # # ----------------------------logging------------------------
    # setlogger(os.path.join(path_logger, 'train.log'))
    # f = open("%s\\train.log" % path_logger, 'w')
    # f.truncate()
    # f.close()
    # ppo_model_train()

    # with open(path_reward_save + '/' + str(1000000) + '_reward_main_9_0.5_0.5_log0.pkl', 'rb') as handle:
    #     (r,mean,var,rc, t_dp_term_star, tau_all_list) = pickle.load(handle)
    # L = 500

#1024000_reward_main_eval_5Gbps_1




    # with open(path_reward_save + '/' + '1024000_reward_main_eval_5Gbps_1.pkl', 'rb') as handle:
    #     (r,mean,var,rc, t_dp_term_star, tau_all_list) = pickle.load(handle)
    # L = 512*2
    #
    # r = np.array(r)
    # rc = np.array(rc)
    # r_mean_2000=mean
    # r_mean_L=[]
    # r_var_L = []
    # for i in range(int(np.floor(len(r)/L))):
    #     r_mean_L.append(r[i*L:i*L+L].mean())
    #     r_var_L.append(r[i*L:i*L + L].var())
    # r_1 = r_mean_L[:750]
    # plt.plot(r_1)
    #
    # with open(path_reward_save + '/' + '768000_reward_main_eval_5Gbps_A2C.pkl', 'rb') as handle:
    #     (r,mean,var,rc, t_dp_term_star, tau_all_list) = pickle.load(handle)
    # # L = 512*4
    #
    # r = np.array(r)
    # rc = np.array(rc)
    # r_mean_2000=mean
    # r_mean_L=[]
    # r_var_L = []
    # for i in range(int(np.floor(len(r)/L))):
    #     r_mean_L.append(r[i*L:i*L+L].mean())
    #     r_var_L.append(r[i*L:i*L + L].var())
    # plt.plot(r_mean_L[:750])
    # r_A2C = r_mean_L
    #
    # with open(path_reward_save + '/' + '768000_reward_main_eval_5Gbps_SAC.pkl', 'rb') as handle:
    #     (r,mean,var,rc, t_dp_term_star, tau_all_list) = pickle.load(handle)
    # # L = 512*4
    #
    # r = np.array(r)
    # rc = np.array(rc)
    # r_mean_2000=mean
    # r_mean_L=[]
    # r_var_L = []
    # for i in range(int(np.floor(len(r)/L))):
    #     r_mean_L.append(r[i*L:i*L+L].mean())
    #     r_var_L.append(r[i*L:i*L + L].var())
    # plt.plot(r_mean_L[:750])
    #
    # r_SAC=r_mean_L
    #
    # plt.title('1500episodes_reward_main_eval_5Gbps_1.pkl')
    # plt.ylim(-0.6,0)
    #
    # plt.show()
    #
    # path_reward_plt_data = os.path.join(root, 'reward_plt_data')
    # if not os.path.exists("%s" % path_reward_plt_data):
    #     os.makedirs("%s" % path_reward_plt_data)
    # # 字典中的key值即为csv中列名
    # df = pd.DataFrame({'DeepMIC': r_1})
    #
    # # 将DataFrame存储为csv,index表示是否显示行名，default=True;
    # # 在参数header和index中使用True或False指定是否指定header（列名，pandas.DataFrame的列）和index（行名，pandas.DataFrame的索引）。默认值为True。
    # df.to_csv(path_reward_plt_data + '/' +'drl_plt_data_DeepMIC.csv', header=True, index=False, sep='\t',float_format='%.5f')  # index为true会多写一列index,分隔符默认为逗号
    #
    # df = pd.DataFrame({'A2C': r_A2C})
    # df.to_csv(path_reward_plt_data + '/' + 'drl_plt_data_A2C.csv', header=True, index=False, sep='\t',
    #           float_format='%.5f')
    #
    # df = pd.DataFrame({'SAC': r_SAC})
    # df.to_csv(path_reward_plt_data + '/' + 'drl_plt_data_SAC.csv', header=True, index=False, sep='\t',
    #           float_format='%.5f')
    #
    # print('Plt End')

    with open(path_reward_save + '/' + str(512000) + '_' + 'GEANT' + '_rewards_marl_env.pkl', 'rb') as handle:
        reward_k1_list, reward_k2_list, reward_k3_list, reward_g_list, g_done_list = pickle.load(handle)

    # with open(path_reward_save + '/' + 'GEANT' + '_rewards_temper_marl_env.pkl', 'rb') as handle:
    #     reward_k1_list, reward_k2_list, reward_k3_list, reward_g_list, g_done_list = pickle.load(handle)

    L = 512  # 512*4

    r = np.array(reward_k1_list)
    r_mean_L=[]
    r_var_L = []
    for i in range(int(np.floor(len(r)/L))):
        r_mean_L.append(r[i*L:i*L+L].mean())
        r_var_L.append(r[i*L:i*L + L].var())

    plt.plot(r_mean_L[:])
    plt.title('reward_k1_list r_mean_L')
    plt.ylim(0, 1)
    plt.show()

    # plt.title('reward_k1_list r_var_L')
    # plt.plot(r_var_L[:])
    # plt.ylim(0, 1)
    # plt.show()

    # 生成成功率
    bool_array = np.array(g_done_list, dtype=bool)
    int_array = bool_array.astype(int)
    g_r = np.array(int_array)

    rr = []
    L_mean = 1000 # 512
    for i in range(int(np.ceil(len(g_done_list)/L_mean))):
        arr = g_r[i * L_mean:i * L_mean + L_mean]
        rr.append((len(arr) - np.sum(arr == 0.0)) / len(arr))
    plt.title('rr')
    plt.plot(rr[:])
    plt.ylim(0, 1)
    plt.show()


    with open(path_reward_save + '/' + '768000_reward_main_eval_5Gbps_TD3.pkl', 'rb') as handle:
        (r,mean,var,rc, t_dp_term_star, tau_all_list) = pickle.load(handle)

    L = 512*4

    r = np.array(r)
    rc = np.array(rc)
    r_mean_2000=mean
    r_mean_L=[]
    r_var_L = []
    for i in range(int(np.floor(len(r)/L))):
        r_mean_L.append(r[i*L:i*L+L].mean())
        r_var_L.append(r[i*L:i*L + L].var())
    plt.plot(r_mean_L[:750])

    plt.title('1500episodes_reward_main_eval_5Gbps_DDPG.pkl')
    plt.ylim(-0.6,0)
    plt.show()
    plt.title('1500episodes_reward_main_eval_5Gbps_DDPG.pkl')
    plt.plot(r_var_L[:750])
    plt.ylim(0, 0.4)
    plt.show()


    rc_mean_L=[]
    rc_var_L = []
    for i in range(int(np.floor(len(rc)/L))):
        rc_mean_L.append(rc[i*L:i*L+L].mean())
        rc_var_L.append(rc[i*L:i*L + L].var())
    plt.plot(rc_mean_L[:1500])
    plt.title('1500episodes_reward_main_eval_5Gbps_DDPG.pkl')
    plt.ylim(0.1, 1)
    plt.show()
    plt.plot(rc_var_L[:1500])
    plt.title('1500episodes_reward_main_eval_5Gbps_DDPG.pkl')
    plt.ylim(0, 0.4)
    plt.show()
    a = 0


    path_reward_plt_data = os.path.join(root, 'reward_plt_data')
    if not os.path.exists("%s" % path_reward_plt_data):
        os.makedirs("%s" % path_reward_plt_data)
    # 字典中的key值即为csv中列名
    df = pd.DataFrame({'DeepMIC': r_mean_L[:1500]})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True;
    # 在参数header和index中使用True或False指定是否指定header（列名，pandas.DataFrame的列）和index（行名，pandas.DataFrame的索引）。默认值为True。
    df.to_csv(path_reward_plt_data + '/' +'drl_plt_data_DeepMIC.csv', header=True, index=False, sep='\t',float_format='%.5f')  # index为true会多写一列index,分隔符默认为逗号

    print('Plt End')
    # model_parameters = filter(lambda p: p.requires_grad, self.mlp_extractor.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print('The network has {} params.'.format(params))

    # rc1_mean_L=[]
    # rc1_var_L = []
    # rc1 = np.clip(r, 2 * r_mean_2000, 0)
    # rc1 = _maxmin_Normalization(rc1, max=0, min=r_mean_2000)
    # for i in range(int(np.floor(len(rc1)/L))):
    #     rc1_mean_L.append(rc1[i*L:i*L+L].mean())
    #     rc1_var_L.append(rc1[i*L:i*L + L].var())
    # plt.plot(rc1_mean_L)
    # plt.title('main_6_rc_mean_L')
    # plt.show()
    # plt.plot(rc1_var_L)
    # plt.title('main_6_rc_var_L')
    # plt.show()



