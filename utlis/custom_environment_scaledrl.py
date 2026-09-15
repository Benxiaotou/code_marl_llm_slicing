#! /usr/bin/env python

import os
import shutil
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import utlis.finetune_generator as fg
import argparse
import pickle
import logging
import time
import collections
import heapq
import random
import matplotlib.pyplot as plt
import math
import collections
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from PIL import Image
from collections import deque
import networkx as nx
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.preprocessing import get_action_dim  # , is_image_space, maybe_transpose, preprocess_obs
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from absl import app
from absl import flags
import seaborn as sns
import pulp
from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value, GLPK

import utlis.network_env
from utlis import game
from utlis.general_function import setlogger, setup_seed
from utlis.config import get_config
from utlis.models import CustomActorCriticPolicy
from utlis.network_env import AbileneTopology, AbileneTraffic, \
    GEANTTopology, GEANTTraffic

OBJ_EPSILON = 1e-12

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_agents', 20, 'number of agents')
flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseline')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations each agent would run')

'''展开 # 展开当前代码块ctrl =  # 彻底展开当前代码块ctrl alt =  # 展开所有代码块ctrl shift +
折叠 # 折叠当前代码块ctrl -  # 彻底折叠当前代码块ctrl alt -  # 折叠所有代码块ctrl shift - '''


class CustomEnv(gym.Env):

    def __init__(self, path_reward_save, link_num_ratio, flow_num_ratio, is_training=True):

        logging.info("Create and initialize the env")
        self.total_steps = None

        config = get_config(FLAGS) or FLAGS
        self.num_steps_per_episode = config.num_steps_per_episode
        self.data_dir = config.data_root + '/'
        self.path_reward_save = path_reward_save

        if config.topology_file == 'Abilene':  # *#
            self.topology = AbileneTopology(config, self.data_dir)
            self.traffic = AbileneTraffic(config, self.topology.num_nodes, self.data_dir, is_training=is_training)
        elif config.topology_file == 'GEANT':  # *#
            self.topology = GEANTTopology(config, self.data_dir)
            self.traffic = GEANTTraffic(config, self.topology, self.data_dir, is_training=is_training)
        elif config.topology_file == 'Xspedius':  # *#
            self.topology = utlis.network_env.XspediusTopology(config, self.data_dir)
            self.traffic = utlis.network_env.XspediusTraffic(config, self.topology, self.data_dir,
                                                             is_training=is_training)
        elif config.topology_file == 'SwitchL3':  # *#
            self.topology = utlis.network_env.SwitchL3Topology(config, self.data_dir)
            self.traffic = utlis.network_env.SwitchL3Traffic(config, self.topology, self.data_dir,
                                                             is_training=is_training)
        elif config.topology_file == 'Uunet':  # *#
            self.topology = utlis.network_env.UunetTopology(config, self.data_dir)
            self.traffic = utlis.network_env.UunetTraffic(config, self.topology, self.data_dir,
                                                          is_training=is_training)
        elif config.topology_file == 'Dfn':
            self.topology = utlis.network_env.DfnTopology(config, self.data_dir)
            self.traffic = utlis.network_env.DfnTraffic(config, self.topology, self.data_dir,
                                                        is_training=is_training)
        elif config.topology_file == 'BtNorthAmerica':
            self.topology = utlis.network_env.BtNorthAmericaTopology(config, self.data_dir)
            pass
        elif config.topology_file == 'Cernet':
            self.topology = utlis.network_env.CernetTopology(config, self.data_dir)
            pass
        elif config.topology_file == 'Chinanet':
            self.topology = utlis.network_env.ChinanetTopology(config, self.data_dir)
            pass

        else:
            pass

        self.topology_file = config.topology_file
        if self.topology_file == 'Abilene':
            self.traffic_matrices = self.traffic.traffic_matrices * 100 * 8 / 300 / 1000  # 转为单位kbps ## Unit: (100 bytes / 5 minutes)  // 100 is the pkt sampling rate
        elif self.topology_file == 'GEANT':
            self.traffic_matrices = self.traffic.traffic_matrices  # 原始数据集单位kbps，<unit type="bandwidth" value="kbps"/>
        elif self.topology_file == 'Xspedius':
            self.traffic_matrices = self.traffic.traffic_matrices
        elif self.topology_file == 'SwitchL3':
            self.traffic_matrices = self.traffic.traffic_matrices
        elif self.topology_file == 'Uunet':
            self.traffic_matrices = self.traffic.traffic_matrices
        else:
            pass

        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = self.traffic.tm_cnt
        self.traffic_file = self.traffic.traffic_file
        self.num_pairs = self.topology.num_pairs
        self.pair_idx_to_sd = self.topology.pair_idx_to_sd
        self.pair_sd_to_idx = self.topology.pair_sd_to_idx
        self.num_nodes = self.topology.num_nodes
        self.num_links = self.topology.num_links
        self.link_idx_to_sd = self.topology.link_idx_to_sd
        self.link_sd_to_idx = self.topology.link_sd_to_idx
        self.link_capacities = self.topology.link_capacities  # capacity(kbps)
        self.link_weights = self.topology.link_weights
        self.shortest_paths_node = self.topology.shortest_paths  # paths consist of nodes
        self.shortest_paths_link = self._convert_to_edge_path(self.shortest_paths_node)
        self.DG = self.topology.DG

        self._get_ecmp_next_hops()

        self.flow_pairs = [p for p in range(self.num_pairs)]
        # for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]

        self.load_multiplier = {}

        self.project_name = config.project_name
        self.num_crit_links = int(np.round(
            link_num_ratio * self.num_links))  # Abilene：0.15，GEANT：0.15，Xspedius：0.15，SwitchL3：0.15    #math.ceil(0.15 * self.num_links)  # 10%*N
        self.max_moves = math.ceil(
            flow_num_ratio * self.num_pairs)  # Abilene：0.20, GEANT：0.15，Xspedius：0.15，SwitchL3     #TopK流的K值#10%*N
        # self.action_dim = self.num_pairs
        # self.max_moves = int(self.action_dim * (config.max_moves / 100.))  ##10%*N*(N-1)
        # assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)
        # self.tm_history = 1
        self.tm_indexes = np.arange(0, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)

        if config.method == 'pure_policy':
            self.baseline = {}

        self.ecmp_mlu_links_util_save_file = self.data_dir + self.topology_file + '/' + self.topology_file + '_ecmp_mlu_links_util_save.pkl'
        if os.path.exists(self.ecmp_mlu_links_util_save_file):
            with open(self.ecmp_mlu_links_util_save_file, 'rb') as pkl:
                self.ecmp_mlu_save, self.ecmp_links_util_save = pickle.load(pkl)
            assert len(self.ecmp_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
        else:
            self.ecmp_mlu_save = []
            self.ecmp_links_util_save = []

        self.optimal_mlu_save_file = self.data_dir + self.topology_file + '/' + self.topology_file + '_optimal_mlu_save.pkl'
        if os.path.exists(self.optimal_mlu_save_file):
            with open(self.optimal_mlu_save_file, 'rb') as pkl:
                self.optimal_mlu_save, self.optimal_mlu_delay_save = pickle.load(pkl)
            assert len(self.optimal_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
        else:
            self.optimal_mlu_save = []
            self.optimal_mlu_delay_save = []

        self._generate_inputs(norm_type='MaxMin')
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)
        print('Number of critical links :', self.num_crit_links)

        # ##****##
        # start_time = time.time()
        # _, solution = self._optimal_routing_mlu(tm_idx=0)
        # end_time = time.time()
        # running_time = end_time - start_time
        # print("测试代码'optimal_routing_mlu_critical_pairs'的运行时间：", running_time * 1000, "ms")
        # ##**##

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_crit_links + 0,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(1, self.num_nodes, self.num_nodes),
                                            dtype=np.float32)

    def reset(self):

        logging.info("************ Reset the env once ************")
        self.tm_idx = 0
        tm = self.traffic_matrices[self.tm_idx]

        # self.state_norm_var = None
        # self.state_norm_mean = None
        # self.state_norm_max = None
        # self.state_norm_min = None

        # self.idx_crit_links, self.idx_crit_flows = self._get_crit_links_flows(tm_idx=self.tm_idx, num_crit_links=self.num_crit_links)
        # self.idx_crit_links = np.random.randint(0, self.num_crit_links, size=self.num_crit_links)#随机初始化关键链路
        ###** 链路挑选方式对比实验 **要注意同时修改reset,step末尾和eval_env_step相关部分##
        ###** 对比1：Random selection
        # self.idx_crit_links = np.random.randint(0, self.num_crit_links, size=self.num_crit_links)  # 随机初始化关键链路

        # ###** 对比2：Node-based selection
        # self.idx_crit_links = self._node_based_selection()

        ###** 对比3：Centrality-based selection
        self.idx_crit_links = self._centrality_based_selection()##**scaledrl


        assert len(
            self.idx_crit_links) == self.num_crit_links, "self.idx_crit_links is not equal to self.num_crit_links"

        # 归一化后的 当前时间步的流量矩阵TM_t， GEANT 维度：(1, 23, 23)
        TM_t = np.array(self.normalized_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
        # 当前时间步的流量矩阵TM_t与上一时间步的流量矩阵TM_(t-1)之差 DTM_t = TM_t - TM_(t-1)，再， GEANT 维度：(1, 23, 23)
        # DTM_t = np.array(self.diff_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
        # # 邻接矩阵AM， GEANT 维度：(1, 23, 23)
        # # AM = nx.to_numpy_array(self.DG, weight=None, nodelist=np.arange(0, self.num_nodes))[np.newaxis, :]
        # # # 关键links信息矩阵 CLIM, GEANT 维度：(1, 23, 23)
        # CLIM = np.zeros((self.num_nodes, self.num_nodes))[np.newaxis, :]
        # for i in self.idx_crit_links:
        #     s, d = self.link_idx_to_sd[i]
        #     CLIM[0][s, d] = 1

        # self.state GEANT 维度：(4, 23, 23)
        self.state = TM_t#np.concatenate((TM_t, DTM_t, CLIM), axis=0)  # 拼接##**scaledrl
        # 可以考虑对所有输入网络的state用固定的均值和方差进行归一化

        self.reward_scaling = 1 / 0.6
        self.reward_list = []
        self.r_mlu_list = []
        self.optimal_mlu_list = []
        self.optimal_mlu_delay_list = []
        self.crit_links_idx_list = []
        self.crit_links_idx_bidir_list = []
        self.time_list = []

        self.step_cnt = 0
        return self.state

    def step(self, action):  # 输出层使用 tanh，即输出为 -1～1 之间的浮点数; 经过缩放，对应到实际动作

        logging.info(
            "Step the env once: 'step_cnt' is %d and the current index 'tm_idx' is %d" % (self.step_cnt, self.tm_idx))
        eval_delay = False

        assert len(action) == self.num_crit_links + 0, "Error from action"
        action_CL = (action + 1) / 2  # 缩放到[0, 1]

        # ECMP对比
        if not os.path.exists(self.ecmp_mlu_links_util_save_file):
            if self.step_cnt + 1 <= self.tm_cnt:
                ecmp_mlu, ecmp_delay, ecmp_links_util = self._eval_ecmp_traffic_distribution(self.tm_idx,
                                                                                             eval_delay=eval_delay)
                self.ecmp_mlu_save.append(ecmp_mlu)
                self.ecmp_links_util_save.append(ecmp_links_util)
            elif self.step_cnt == self.tm_cnt:
                with open(self.ecmp_mlu_links_util_save_file, 'wb') as handle:
                    pickle.dump((self.ecmp_mlu_save, self.ecmp_links_util_save), handle)
                ecmp_mlu = self.ecmp_mlu_save[self.tm_idx]
                ecmp_links_util = self.ecmp_links_util_save[self.tm_idx]
        else:
            ecmp_mlu = self.ecmp_mlu_save[self.tm_idx]
            ecmp_links_util = self.ecmp_links_util_save[self.tm_idx]

        mlu, drl_links_util = self._eval_crit_links_and_crit_flows(self.tm_idx, action_CL)

        # start_time = time.time()
        # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, action_CF)
        # end_time = time.time()
        # running_time = end_time - start_time
        # print("代码'optimal_routing_mlu_critical_pairs'的运行时间：", running_time*1000, "ms")
        # mlu, delay = self._eval_critical_flow_and_ecmp(self.tm_idx, action_CF, solution, eval_delay=eval_delay)

        # crit_topk = self._get_critical_topK_flows(self.tm_idx)
        # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, crit_topk)
        # crit_mlu, crit_delay = self._eval_critical_flow_and_ecmp(self.tm_idx, crit_topk, solution, eval_delay=eval_delay)
        #
        # topk = self._get_topK_flows(self.tm_idx, self.lp_pairs)
        # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, topk)
        # topk_mlu, topk_delay = self._eval_critical_flow_and_ecmp(self.tm_idx, topk, solution, eval_delay=eval_delay)

        # _, solution = self._optimal_routing_mlu(self.tm_idx)
        # optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution, eval_delay=eval_delay)

        if not os.path.exists(self.optimal_mlu_save_file):
            if self.step_cnt + 1 <= self.tm_cnt:
                _, solution = self._optimal_routing_mlu(self.tm_idx)
                optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution,
                                                                                eval_delay=True)
                self.optimal_mlu_save.append(optimal_mlu)
                self.optimal_mlu_delay_save.append(optimal_mlu_delay)
            elif self.step_cnt == self.tm_cnt:
                with open(self.optimal_mlu_save_file, 'wb') as handle:
                    pickle.dump((self.optimal_mlu_save, self.optimal_mlu_delay_save), handle)
                optimal_mlu = self.optimal_mlu_save[self.tm_idx]
                optimal_mlu_delay = self.optimal_mlu_delay_save[self.tm_idx]
        else:
            optimal_mlu = self.optimal_mlu_save[self.tm_idx]
            optimal_mlu_delay = self.optimal_mlu_delay_save[self.tm_idx]

        norm_mlu = optimal_mlu / mlu
        line = 'im_idx=' + str(self.tm_idx) + ', norm_mlu=' + str(np.around(norm_mlu, decimals=6)) \
               + ', mlu=' + str(np.around(mlu, decimals=6))

        # norm_crit_mlu = optimal_mlu / crit_mlu
        # line += ', norm_crit_mlu=' + str(np.around(norm_crit_mlu, decimals=6)) + ', crit_mlu=' + str(np.around(crit_mlu, decimals=6))
        #
        # norm_topk_mlu = optimal_mlu / topk_mlu
        # line += ', norm_topk_mlu=' + str(np.around(norm_topk_mlu, decimals=6)) + ', topk_mlu=' + str(np.around(topk_mlu, decimals=6))

        norm_ecmp_mlu = optimal_mlu / ecmp_mlu
        line += ', norm_ecmp_mlu=' + str(np.around(norm_ecmp_mlu, decimals=6)) + ', ecmp_mlu=' + str(
            np.around(ecmp_mlu, decimals=6)) + ', '

        # if eval_delay:
        #     solution = self._optimal_routing_delay(self.tm_idx)
        #     optimal_delay = self._eval_optimal_routing_delay(self.tm_idx, solution)
        #
        #     line += str(optimal_delay / delay) + ', '
        #     line += str(optimal_delay / crit_delay) + ', '
        #     line += str(optimal_delay / topk_delay) + ', '
        #     line += str(optimal_delay / optimal_mlu_delay) + ', '
        #     if ecmp:
        #         line += str(optimal_delay / ecmp_delay) + ', '
        #
        #     assert self.tm_idx in self.load_multiplier, (self.tm_idx)
        #     line += str(self.load_multiplier[self.tm_idx]) + ', '
        line = "step_cnt=" + str(self.step_cnt + 1) + ', ' + line
        print(line[:-2])

        # *********计算 reward **********
        reward = (norm_mlu - 0.4) * self.reward_scaling
        print('reward=%0.6f' % reward)

        self.reward_list.append(reward)
        self.r_mlu_list.append(mlu)
        self.optimal_mlu_list.append(optimal_mlu)
        if self.step_cnt + 1 == self.total_steps:
            with open(self.path_reward_save + '/' + str(self.step_cnt) + self.topology_file + '_reward_main_XX.pkl',
                      'wb') as handle:
                pickle.dump((self.reward_list, self.r_mlu_list, self.optimal_mlu_list), handle)

        # *********更新 state **********
        self.tm_idx = self.tm_idx + 1 if self.tm_idx + 1 < self.tm_cnt else 0

        # self.idx_crit_links, self.idx_crit_flows = self._get_crit_links_flows(tm_idx=self.tm_idx, num_crit_links=self.num_crit_links)
        # self.idx_crit_links = [62, 60, 32, 28, 52, 41, 22, 19, 51, 20, 39]#[32, 60, 28, 62, 19, 52, 41, 22, 20, 39, 51]
        # self.idx_crit_links = self._get_crit_links(ecmp_links_util=ecmp_links_util,
        #                                            num_rollout_step=int(self.tm_cnt * 0.6),# !!! 为1时即可得到默认的关键links
        #                                            update_interval=1)

        # num_rollout_step = int(self.tm_cnt * 0.8)#*
        # self.idx_crit_links = self._get_crit_links(ecmp_links_util=ecmp_links_util,
        #                                            num_rollout_step=num_rollout_step,  # !!! 为1时即可得到默认的关键links
        #                                            offset=int(num_rollout_step * 0.6),
        #                                            scale=int(num_rollout_step * 0.2),
        #                                            update_interval=1)##**scaledrl

        ###** 链路挑选方式对比实验 **要注意同时修改reset,step末尾和eval_env_step相关部分##
        ###** 对比1，对比2，对比3均为离线链路选择算法，因次reset后不需要更新

        assert len(
            self.idx_crit_links) == self.num_crit_links, "self.idx_crit_links is not equal to self.num_crit_links"

        # 归一化后的 当前时间步的流量矩阵TM_t， GEANT 维度：(1, 23, 23)
        TM_t = np.array(self.normalized_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
        # 当前时间步的流量矩阵TM_t与上一时间步的流量矩阵TM_(t-1)之差 DTM_t = TM_t - TM_(t-1)，再， GEANT 维度：(1, 23, 23)
        # DTM_t = np.array(self.diff_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
        # # 邻接矩阵AM， GEANT 维度：(1, 23, 23)
        # # AM = nx.to_numpy_array(self.DG, weight=None, nodelist=np.arange(0, self.num_nodes))[np.newaxis, :]
        # # # 关键links信息矩阵 CLIM, GEANT 维度：(1, 23, 23)
        # CLIM = np.zeros((self.num_nodes, self.num_nodes))[np.newaxis, :]
        # for i in self.idx_crit_links:
        #     s, d = self.link_idx_to_sd[i]
        #     CLIM[0][s, d] = 1

        # self.state GEANT 维度：(4, 23, 23)
        self.state = TM_t#np.concatenate((TM_t, DTM_t, CLIM), axis=0)  # 拼接##**scaledrl
        # 可以考虑对所有输入网络的state用固定的均值和方差进行归一化

        # done = True if (self.step_cnt+1) % int(self.num_steps_per_episode) == 0 else False
        done = False
        info = {}

        self.step_cnt += 1

        return self.state, reward, done, info
        ## self.state为array类型('numpy.ndarray')，reward为float类型，done为False(episode未终止)或True（episode终止）

    # ########分析双向链路，从技术文件可看出链路的两个不同方向链路容量相同并且OSPF的cost权重也相同
    # def step(self, action):
    #
    #     logging.info("Step the env once and the current index 'tm_idx' is %d" % self.tm_idx)
    #     ecmp = True
    #     eval_delay = False
    #
    #     action_CF = action.argsort()[-self.max_moves:]
    #
    #     # if ecmp:
    #     #     ecmp_mlu, ecmp_delay = self._eval_ecmp_traffic_distribution(self.tm_idx, eval_delay=eval_delay)
    #
    #     print("step_cnt=%d" % self.step_cnt)
    #
    #     #拥塞链路挑选
    #     num_critical_links = 8
    #     link_loads = self._ecmp_traffic_distribution(self.tm_idx)
    #     critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:num_critical_links]#np.argsort函数元素从小到大排列（升序）
    #     self.crit_links_idx_list += critical_link_indexes.tolist()
    #
    #
    #
    #
    #
    #     # self.pair_idx_to_sd, self.pair_sd_to_idx,self.link_idx_to_sd, self.link_sd_to_idx
    #     self.link_ids_to_idd = {}
    #     self.link_idd_to_ids = {}
    #     idx_data = 0
    #     for i in range(self.num_links):
    #
    #         s, d = self.link_idx_to_sd[i]
    #         if s<d:
    #             self.link_ids_to_idd[i] = idx_data#[idx_data, (s, d), (d, s)]
    #             # self.link_idd_to_ids[idx_data] = [i]
    #             assert (d,s) in self.link_sd_to_idx, "Error, 该链路单向？"
    #             idx_data += 1
    #         else:
    #             self.link_ids_to_idd[i] = self.link_ids_to_idd[self.link_sd_to_idx[(d, s)]]
    #             # self.link_idd_to_ids[self.link_sd_to_idx[(d, s)]] += [i]
    #
    #     bidir_link_loads = np.zeros((int(self.num_links/2)))
    #     for i in range(self.num_links):
    #         bidir_link_loads[self.link_ids_to_idd[i]] = link_loads[i]
    #     num_critical_links = 5
    #     critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:num_critical_links]  # np.argsort函数元素从小到大排列（升序）
    #     self.crit_links_idx_list += critical_link_indexes.tolist()
    #
    #
    #
    #
    #         # 画出基于一个流量矩阵下的各个链路的利用率
    #     lu = link_loads / self.link_capacities
    #     plt.bar(np.arange(0, lu.__len__()), lu)  # 创建柱状图
    #     # plt.rcParams['font.sans-serif'] = ['SimHei']
    #     plt.title('Link utilization for the traffic matrix (tm_idx=%d) of Abilene' % self.tm_idx)
    #     plt.xlabel('Link index')
    #     plt.ylabel('Link utilization')
    #     plt.show()
    #
    #
    #     # 画出基于一个流量矩阵下的
    #     self.dict_link_tm_demand = {}#
    #     dict_link_tm_demand = self._e_t_d_rewrite(self.tm_idx)
    #     for c_l_idx in self.crit_links_idx_list:
    #
    #         # 混淆矩阵的数据
    #         arr_link_tm=dict_link_tm_demand[c_l_idx]
    #         # 在索引位置（s=d）插入新的数值 0
    #         for ist in range(self.num_nodes):
    #             arr_link_tm = np.insert(arr_link_tm, ist+ist*self.num_nodes, 0)
    #
    #         assert np.round(link_loads[c_l_idx], 4) == np.round(arr_link_tm.sum(), 4), "Error"
    #         # 归一化，数值代表每个流的流量占总总流量的百分比
    #         arr_link_tm = np.round(arr_link_tm/arr_link_tm.sum(), 3)
    #         #reshape整形成类似混淆矩阵的形式
    #         arr_link_tm_reshape = arr_link_tm.reshape((self.num_nodes, self.num_nodes))
    #
    #         confusion_matrix_data = arr_link_tm_reshape.tolist()
    #         # 绘制混淆矩阵图
    #         plt.figure(figsize=(8, 7))
    #         sns.heatmap(confusion_matrix_data, annot=True, cmap="Blues")
    #         plt.xlabel("Source node")
    #         plt.ylabel("Destination node")
    #         plt.title("Congestion link "+str(c_l_idx))
    #         plt.show()
    #
    #     self.dict_link_tm_demand = {}
    #
    #
    #
    #     if self.step_cnt + 1 == self.tm_cnt:
    #         arr = np.array(self.crit_links_idx_list)
    #         unique_elements, counts = np.unique(arr, return_counts=True)
    #
    #         # with open(p_cl + '/'+ self.topology_file+ '_top' +str(num_critical_links)+'_crit_links.pkl', 'rb') as handle:
    #         #     arr, unique_elements, counts = pickle.load(handle)
    #
    #         x = unique_elements
    #         y = counts
    #         plt.bar(x, y)  # 创建柱状图
    #         # plt.rcParams['font.sans-serif'] = ['SimHei']
    #         plt.title('Number of link congestion (top %d) in traffic matrices of Abilene' % num_critical_links)
    #         plt.xlabel('Link_index')
    #         plt.ylabel('Counts of occurrences')
    #         plt.show()
    #
    #         p_cl = self.path_reward_save[:-11] + 'result_crit_links_idx'
    #         with open(p_cl + '/'+ self.topology_file+ '_top' +str(num_critical_links)+'_crit_links.pkl', 'wb') as handle:
    #             pickle.dump((arr, unique_elements, counts), handle)
    #
    #         print('Stop')
    #
    #     else:
    #         pass
    #
    #     reward = 0.1
    #
    #     self.state = np.array(self.normalized_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis,:]
    #     #可以对所有输入网络的state用固定的均值和方差进行归一化
    #
    #     # done = True if (self.step_cnt+1) % int(self.num_steps_per_episode) == 0 else False
    #     done = False
    #     info = {}
    #
    #     self.tm_idx = self.tm_idx + 1 if self.tm_idx+1 < self.tm_cnt else 0
    #     self.step_cnt += 1
    #
    #     return self.state, reward, done, info
    #     ## self.state为array类型('numpy.ndarray')，reward为float类型，done为False(episode未终止)或True（episode终止）

    ################单向链路分析#################！！
    # def step(self, action):
    #
    #     logging.info("Step the env once and the current index 'tm_idx' is %d" % self.tm_idx)
    #     ecmp = True
    #     eval_delay = False
    #
    #     action_CF = action.argsort()[-self.max_moves:]
    #
    #     # if ecmp:
    #     #     ecmp_mlu, ecmp_delay = self._eval_ecmp_traffic_distribution(self.tm_idx, eval_delay=eval_delay)
    #
    #     print("step_cnt=%d" % self.step_cnt)
    #
    #     num_critical_links = 8#10
    #     link_loads = self._ecmp_traffic_distribution(self.tm_idx)
    #     critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:num_critical_links]#np.argsort函数元素从小到大排列（升序）
    #     self.crit_links_idx_list += critical_link_indexes.tolist()
    #
    #     if self.tm_idx == 200:#self.tm_idx == 200第201个流量矩阵
    #         print('Stop debugger and the tm_idx is %d' % self.tm_idx)
    #
    #     # ## 画出基于一个流量矩阵下的各个链路的利用率
    #     # lu = link_loads / self.link_capacities
    #     # plt.bar(np.arange(0, lu.__len__()), lu)  # 创建柱状图
    #     # # plt.rcParams['font.sans-serif'] = ['SimHei']
    #     # plt.title('Link utilization for the traffic matrix (tm_idx=%d) of Abilene' % self.tm_idx)
    #     # plt.xlabel('Link index')
    #     # plt.ylabel('Link utilization')
    #     # plt.show()
    #
    #
    #     ## 以混淆矩阵的形式画出基于一个流量矩阵下的一系列‘源->目的地对流’占链路总流量的百分比
    #     self.dict_link_tm_demand = {} #
    #     dict_link_tm_demand = self._e_t_d_rewrite(self.tm_idx)
    #     for c_l_idx in self.crit_links_idx_list:
    #         # self.pair_idx_to_sd, self.pair_sd_to_idx,self.link_idx_to_sd, self.link_sd_to_idx
    #         # 混淆矩阵的数据
    #         arr_link_tm=dict_link_tm_demand[c_l_idx]
    #         # 在索引位置（s=d）插入新的数值 0
    #         for ist in range(self.num_nodes):
    #             arr_link_tm = np.insert(arr_link_tm, ist+ist*self.num_nodes, 0)
    #
    #         assert np.round(link_loads[c_l_idx], 4) == np.round(arr_link_tm.sum(), 4), "Error"
    #         # 归一化，数值代表每个流的流量占总流量的百分比
    #         arr_link_tm = np.round(arr_link_tm/arr_link_tm.sum(), 3)
    #         #reshape整形成类似混淆矩阵的形式
    #         arr_link_tm_reshape = arr_link_tm.reshape((self.num_nodes, self.num_nodes))
    #
    #         confusion_matrix_data = arr_link_tm_reshape.tolist()
    #         # 绘制混淆矩阵图
    #         # plt.figure(figsize=(8, 7))## Abilene
    #         plt.figure(figsize=(14, 10))##GEANT
    #         sns.heatmap(confusion_matrix_data, annot=True, cmap="Blues")
    #         plt.xlabel("Source node")
    #         plt.ylabel("Destination node")
    #         plt.title("Congestion link %d of tm_%d" % (c_l_idx, self.tm_idx))
    #         plt.show()
    #     self.dict_link_tm_demand = {}
    #
    #
    #     #在所有流量矩阵中链路拥塞数(前K条)出现的链路和次数
    #     if self.step_cnt + 1 == self.tm_cnt:
    #         arr = np.array(self.crit_links_idx_list)
    #         unique_elements, counts = np.unique(arr, return_counts=True)
    #
    #         # with open(p_cl + '/'+ self.topology_file+ '_top' +str(num_critical_links)+'_crit_links.pkl', 'rb') as handle:
    #         #     arr, unique_elements, counts = pickle.load(handle)
    #
    #         x = unique_elements
    #         y = counts
    #         plt.bar(x, y)  # 创建柱状图
    #         # plt.rcParams['font.sans-serif'] = ['SimHei']
    #         plt.title('Number of link congestion (top %d) in traffic matrices of Abilene' % num_critical_links)
    #         plt.xlabel('Link_index')
    #         plt.ylabel('Counts of occurrences')
    #         plt.show()
    #
    #         p_cl = self.path_reward_save[:-11] + 'result_crit_links_idx'
    #         with open(p_cl + '/'+ self.topology_file+ '_top' +str(num_critical_links)+'_crit_links.pkl', 'wb') as handle:
    #             pickle.dump((arr, unique_elements, counts), handle)
    #
    #         print('link congestion (top %d): ' % num_critical_links + ' '.join([str(x) for x in unique_elements]))
    #         print('The count of the link congestion: ' + ' '.join([str(x) for x in counts]))
    #         print('Stop')
    #     else:
    #         pass
    #
    #
    #     reward = 0.1
    #
    #     idx_offset = self.tm_history - 1
    #     self.state = np.array(self.normalized_traffic_matrices[self.tm_idx-idx_offset], dtype=np.float32)[np.newaxis,:]
    #     #可以对所有输入网络的state用固定的均值和方差进行归一化
    #
    #     # done = True if (self.step_cnt+1) % int(self.num_steps_per_episode) == 0 else False
    #     done = False
    #     info = {}
    #
    #     self.tm_idx = self.tm_idx + 1 if self.tm_idx+1 < self.tm_cnt else 0
    #     self.step_cnt += 1
    #
    #     return self.state, reward, done, info
    #     ## self.state为array类型('numpy.ndarray')，reward为float类型，done为False(episode未终止)或True（episode终止）

    def _generate_inputs(self, norm_type='MaxMin'):
        self.normalized_traffic_matrices = np.zeros(
            (self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2]),
            dtype=np.float32)  # tm state  [Valid_tms, Node, Node]
        self.diff_traffic_matrices = np.zeros(
            (self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2]), dtype=np.float32)

        for tm_idx in self.tm_indexes:

            if norm_type == 'MaxMin':

                tm_max_element = np.max(self.traffic_matrices[tm_idx])
                self.normalized_traffic_matrices[tm_idx, :, :] = self.traffic_matrices[
                                                                     tm_idx] / tm_max_element  # [Valid_tms, Node, Node, History]

                if tm_idx == 0:
                    self.diff_traffic_matrices[tm_idx, :, :] = np.zeros(
                        (self.traffic_matrices_dims[1], self.traffic_matrices_dims[2]), dtype=np.float32)
                else:
                    self.diff_traffic_matrices[tm_idx, :, :] = 5 * (
                                self.traffic_matrices[tm_idx] - self.traffic_matrices[tm_idx - 1]) \
                                                               / tm_max_element
                    # 注意这一步会使得self.state 的数值小于0，因为两个流量矩阵相减后的数值可以为负。
            else:
                print("The norm_type is wrong")
                # self.normalized_traffic_matrices[tm_idx, :, :] = self.traffic_matrices[tm_idx]  # [Valid_tms, Node, Node, History]

    def _get_topK_flows(self, tm_idx, pairs):
        tm = self.traffic_matrices[tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]

        sorted_f = sorted(f.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        cf = []
        for i in range(self.max_moves):
            cf.append(sorted_f[i][0])

        return cf

    def _get_topK_flows_ScaleDRL(self, tm_idx, pairs, max_moves):
        tm = self.traffic_matrices[tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]

        sorted_f = sorted(f.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        cf = []
        for i in range(max_moves):
            cf.append(sorted_f[i][0])

        return cf

    def _get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    def _ecmp_next_hop_distribution(self, link_loads, demand, src, dst):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)
        # if next_hops_cnt > 1:
        # print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])

        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
            self._ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst)

    def _ecmp_traffic_distribution(self, tm_idx):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self._ecmp_next_hop_distribution(link_loads, demand, s, d)

        return link_loads

    def _get_critical_topK_flows(self, tm_idx, critical_links=5):
        link_loads = self._ecmp_traffic_distribution(tm_idx)
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[
                                :critical_links]  # np.argsort函数元素从小到大排列（升序）

        cf_potential = []
        for pair_idx in range(self.num_pairs):
            for path in self.shortest_paths_link[pair_idx]:
                if len(set(path).intersection(critical_link_indexes)) > 0:
                    cf_potential.append(pair_idx)
                    break

        # print(cf_potential)
        assert len(cf_potential) >= self.max_moves, \
            ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)" % (
                cf_potential, self.max_moves, critical_links))

        return self._get_topK_flows(tm_idx, cf_potential)

    def _eval_ecmp_traffic_distribution(self, tm_idx, eval_delay=False):
        eval_link_loads = self._ecmp_traffic_distribution(tm_idx)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay, eval_link_loads / self.link_capacities

    def _optimal_routing_mlu(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        model = LpProblem(name="routing")

        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name="link_load", indexs=self.links)

        r = LpVariable(name="congestion_ratio")

        for pr in self.lp_pairs:
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1,
                "flow_conservation_constr1_%d" % pr)

        for pr in self.lp_pairs:
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                "flow_conservation_constr2_%d" % pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum(
                        [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                              "flow_conservation_constr3_%d_%d" % (pr, n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr] * ratio[pr, e[0], e[1]] for pr in self.lp_pairs]),
                      "link_load_constr%d" % ei)
            model += (link_load[ei] <= self.link_capacities[ei] * r, "congestion_ratio_constr%d" % ei)

        model += r + OBJ_EPSILON * lpSum([link_load[e] for e in self.links])

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return obj_r, solution

    def _eval_optimal_routing_mlu(self, tm_idx, solution, eval_delay=False):
        optimal_link_loads = np.zeros((self.num_links))
        eval_tm = self.traffic_matrices[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand * solution[i, e[0], e[1]]

        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            optimal_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_max_utilization, delay

    def _optimal_routing_mlu_critical_pairs(self, tm_idx, critical_pairs):
        tm = self.traffic_matrices[tm_idx]

        pairs = critical_pairs

        demands = {}
        background_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            # background link load
            if i not in critical_pairs:
                self._ecmp_next_hop_distribution(background_link_loads, tm[s][d], s, d)
            else:
                demands[i] = tm[s][d]

        model = LpProblem(name="routing")  ##LpProblem() 构造函数还有一个可选参数 sense，用于指定问题的类型。如果省略了这个参数，问题的类型默认为 LpMinimize。

        pair_links = [(pr, e[0], e[1]) for pr in pairs for e in self.lp_links]
        ratio = LpVariable.dicts(name="ratio", indexs=pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name="link_load", indexs=self.links)

        r = LpVariable(name="congestion_ratio")

        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]])
                      - lpSum(
                [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1,
                      "flow_conservation_constr1_%d" % pr)

        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]])
                      - lpSum(
                [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                      "flow_conservation_constr2_%d" % pr)

        for pr in pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n])
                              - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                              "flow_conservation_constr3_%d_%d" % (pr, n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (
            link_load[ei] == background_link_loads[ei] + lpSum([demands[pr] * ratio[pr, e[0], e[1]] for pr in pairs]),
            "link_load_constr%d" % ei)
            model += (link_load[ei] <= self.link_capacities[ei] * r, "congestion_ratio_constr%d" % ei)

        model += r + OBJ_EPSILON * lpSum([link_load[ei] for ei in self.links])

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return obj_r, solution

    def _eval_critical_flow_and_ecmp(self, tm_idx, critical_pairs, solution, eval_delay=False):
        eval_tm = self.traffic_matrices[tm_idx]
        eval_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            if i not in critical_pairs:
                self._ecmp_next_hop_distribution(eval_link_loads, eval_tm[s][d], s, d)
            else:
                demand = eval_tm[s][d]
                for e in self.lp_links:
                    link_idx = self.link_sd_to_idx[e]
                    eval_link_loads[link_idx] += eval_tm[s][d] * solution[i, e[0], e[1]]

        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay

    def _optimal_routing_delay(self, tm_idx):
        assert tm_idx in self.load_multiplier, (tm_idx)
        tm = self.traffic_matrices[tm_idx] * self.load_multiplier[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        model = LpProblem(name="routing")

        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name="link_load", indexs=self.links)

        f = LpVariable.dicts(name="link_cost", indexs=self.links)

        for pr in self.lp_pairs:
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1,
                "flow_conservation_constr1_%d" % pr)

        for pr in self.lp_pairs:
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                "flow_conservation_constr2_%d" % pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum(
                        [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                              "flow_conservation_constr3_%d_%d" % (pr, n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr] * ratio[pr, e[0], e[1]] for pr in self.lp_pairs]),
                      "link_load_constr%d" % ei)
            model += (f[ei] * self.link_capacities[ei] >= link_load[ei], "cost_constr1_%d" % ei)
            model += (f[ei] >= 3 * link_load[ei] / self.link_capacities[ei] - 2 / 3, "cost_constr2_%d" % ei)
            model += (f[ei] >= 10 * link_load[ei] / self.link_capacities[ei] - 16 / 3, "cost_constr3_%d" % ei)
            model += (f[ei] >= 70 * link_load[ei] / self.link_capacities[ei] - 178 / 3, "cost_constr4_%d" % ei)
            model += (f[ei] >= 500 * link_load[ei] / self.link_capacities[ei] - 1468 / 3, "cost_constr5_%d" % ei)
            model += (f[ei] >= 5000 * link_load[ei] / self.link_capacities[ei] - 16318 / 3, "cost_constr6_%d" % ei)

        model += lpSum(f[ei] for ei in self.links)

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return solution

    def _eval_optimal_routing_delay(self, tm_idx, solution):
        optimal_link_loads = np.zeros((self.num_links))
        assert tm_idx in self.load_multiplier, (tm_idx)
        eval_tm = self.traffic_matrices[tm_idx] * self.load_multiplier[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand * solution[i, e[0], e[1]]

        optimal_delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_delay

    def _convert_to_edge_path(self, node_paths):
        edge_paths = []
        num_pairs = len(node_paths)
        for i in range(num_pairs):
            edge_paths.append([])
            num_paths = len(node_paths[i])
            for j in range(num_paths):
                edge_paths[i].append([])
                path_len = len(node_paths[i][j])
                for n in range(path_len - 1):
                    e = self.link_sd_to_idx[(node_paths[i][j][n], node_paths[i][j][n + 1])]
                    assert e >= 0 and e < self.num_links
                    edge_paths[i][j].append(e)

        return edge_paths

    def _maxmin_Normalization(self, x, max, min):
        x = (x - min) / (max - min)
        return x

    # def _get_critical_links(self, tm_idx, num_critical_links=5):
    #     link_loads = self._ecmp_traffic_distribution(tm_idx)
    #     critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:num_critical_links]#np.argsort函数元素从小到大排列（升序）
    #     self.crit_links_idx_list += critical_link_indexes.tolist()
    #     # cf_potential = []
    #     # for pair_idx in range(self.num_pairs):
    #     #     for path in self.shortest_paths_link[pair_idx]:
    #     #         if len(set(path).intersection(critical_link_indexes)) > 0:
    #     #             cf_potential.append(pair_idx)
    #     #             break
    #
    #     # # print(cf_potential)
    #     # assert len(cf_potential) >= self.max_moves, \
    #     #     ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)" % (
    #     #     cf_potential, self.max_moves, critical_links))
    #
    #     # return self._get_topK_flows(tm_idx, cf_potential)

    def _e_t_d_rewrite(self, tm_idx):  # , pair_idx_in_critlink):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]

        for i in range(self.num_links):  #
            self.dict_link_tm_demand[i] = np.zeros(self.num_pairs)  #

        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self._e_n_h_d_rewrite(link_loads, demand, s, d, pair_idx)  #

        return self.dict_link_tm_demand

    def _e_n_h_d_rewrite(self, link_loads, demand, src, dst, pair_idx):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)

        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            link_idx = self.link_sd_to_idx[(src, np)]  #
            self.dict_link_tm_demand[link_idx][pair_idx] += ecmp_demand  #

            link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
            self._e_n_h_d_rewrite(link_loads, ecmp_demand, np, dst, pair_idx)

    def _eval_crit_links_and_crit_flows(self, tm_idx, action_CL):  # action_CL中的值范围[0,1]
        eval_tm = self.traffic_matrices[tm_idx]
        eval_link_loads = np.zeros((self.num_links))
        ##**#start_time
        start_time = time.time()
        ##**##

        idx_crit_flows = self._get_topK_flows_ScaleDRL(self.tm_idx, self.flow_pairs, max_moves=math.ceil(0.5*self.num_pairs))#**#0.5,1  # Top K

        for idx_DF in range(self.num_pairs):
            if idx_DF not in idx_crit_flows:  # 非关键流
                s, d = self.pair_idx_to_sd[idx_DF]
                demand = eval_tm[s][d]
                if demand != 0:
                    self._ecmp_next_hop_distribution(eval_link_loads, demand, s, d)

        virtual_DG = copy.deepcopy(self.DG)
        # 更改link权重信息
        for i in range(len(self.idx_crit_links)):
            idx_CL = self.idx_crit_links[i]
            s, d = self.link_idx_to_sd[idx_CL]
            original_weight = self.DG.edges[s, d]['weight']
            virtual_DG.edges[s, d]['weight'] = 3 * original_weight * action_CL[i]

        # # K_num = 1  # **# scaledrl时为1
        # virtual_K_shortest_paths = {}
        # for idx_CF in range(self.num_pairs):
        #     s, d = self.pair_idx_to_sd[idx_CF]
        #     demand = eval_tm[s][d]
        #     if demand != 0:
        #         # virtual_copy_DG = virtual_DG.copy()
        #         paths_list = []
        #         path = nx.shortest_path(virtual_DG, s, d, weight='weight', method='dijkstra')
        #         paths_list.append(path)
        #         # paths_list = list(nx.all_shortest_paths(virtual_DG, s, d, weight='weight', method='dijkstra'))[:3]
        #         virtual_K_shortest_paths[idx_CF] = paths_list
        #
        #         self._ksp_next_hop_distribution(eval_link_loads, demand, s, d,
        #                                         virtual_K_shortest_paths[idx_CF])
        #
        # eval_max_utilization = np.max(eval_link_loads / self.link_capacities)


        K_num = 2  # **# 2 #Abilene：1，GEANT：2，Xspedius：2，SwitchL3：2
        virtual_K_shortest_paths = {}
        virtual_copy_DG = virtual_DG.copy()
        with open(self.path_reward_save + '/' + 'tempor_buffer_virtual_copy_DG.pkl', 'wb') as handle:
            pickle.dump((virtual_copy_DG), handle)

        for idx_CF in idx_crit_flows:
            s, d = self.pair_idx_to_sd[idx_CF]
            demand = eval_tm[s][d]
            if demand != 0:
                # # *## GEANT考虑采用此种多路径方式（法1）
                # with open(self.path_reward_save + '/' + 'tempor_buffer_virtual_copy_DG.pkl', 'rb') as pkl:
                #     virtual_copy_DG = pickle.load(pkl)
                # paths_list = []
                # path = nx.shortest_path(virtual_copy_DG, s, d, weight='weight', method='dijkstra')
                # paths_list.append(path)
                #
                # for k in range(K_num - 1):  # 疑问：如果nx.shortest_path得到的path为多个，则最终循环后的path数量超过K_num？ 答：该函数应该仅仅返回一个路径（即使是等价多路径，也是返回一个），参见：https://www.osgeo.cn/networkx/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html?highlight=shortest_path
                #     ebunch = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                #     virtual_copy_DG.remove_edges_from(ebunch)
                #     if nx.has_path(virtual_copy_DG, s, d):
                #         path = nx.shortest_path(virtual_copy_DG, s, d, weight='weight', method='dijkstra')
                #         paths_list.append(path)
                #     else:
                #         break  # 不存在K_num条路径，提前返回
                #
                # virtual_K_shortest_paths[idx_CF] = paths_list

                # # ##****## Abilene，Xspedius，SwitchL3 采用它，
                # start_time = time.time()
                # # ##****##

                # 生成K短路径（法2）#当Abilene时应更改K_num
                shortest_paths_generator = nx.shortest_simple_paths(virtual_DG, s, d, weight='weight')
                KSP_list = []
                for counter, path in enumerate(shortest_paths_generator):
                    KSP_list.append(path)
                    if counter == K_num - 1:
                        break
                virtual_K_shortest_paths[idx_CF] = KSP_list

                # # ##****##
                # end_time = time.time()
                # running_time = end_time - start_time
                # print("测试代码'optimal_routing_mlu_critical_pairs'的运行时间：", running_time * 1000, "ms")
                # # ##**##

                self._ksp_next_hop_distribution(eval_link_loads, demand, s, d,
                                                virtual_K_shortest_paths[idx_CF])

        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)


        ##****##end_time
        end_time = time.time()
        running_time = end_time - start_time
        self.time_list.append(running_time*1000)
        print("测试代码的运行时间：", running_time * 1000, "ms")
        # ##**##

        # delay = 0
        # if eval_delay:
        #     assert tm_idx in self.load_multiplier, (tm_idx)
        #     eval_link_loads *= self.load_multiplier[tm_idx]
        #     delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))
        return eval_max_utilization, eval_link_loads / self.link_capacities

    def _ksp_next_hop_distribution(self, link_loads, demand, src, dst,
                                   paths):

        if src == dst:
            return

        # next_hops = virtual_attr_next_hops[0]
        # next_hop_pathnum = virtual_attr_next_hops[1]
        # next_hop_hopnum = virtual_attr_next_hops[2]
        #
        # next_hops = next_hops[src, dst]  # 例 (0, 7): [10, 13]
        # next_hops_cnt = len(next_hops)
        # traffic_demand = []
        #
        # if policy == 'WCMP':
        #     attr_next_hops = [] # (1)出口带宽；(2)下一跳后的路径数；(3)下一跳后的不同路径平均跳数（相同下一跳的多个路径的路由跳数平均值）
        #     beta = np.array([0.5, 0.2, 0.3])  # β1，β2，β3
        #
        #     for n_p in next_hops:
        #         attr_n_l_capacity = self.link_capacities[self.link_sd_to_idx[(src, n_p)]]
        #         attr_n_h_pathnum = next_hop_pathnum[src, dst][n_p]
        #         attr_n_h_averhopnum = np.array(next_hop_hopnum[src, dst][n_p]).mean()
        #         attr_next_hops.append([attr_n_l_capacity, 1/attr_n_h_pathnum, 1/attr_n_h_averhopnum])
        #     # 先每个属性除以最大值，归一化 # 再乘以属性影响因子β1，β2，β3
        #     attr_next_hops = np.array(attr_next_hops)
        #     attr_next_hops = attr_next_hops / attr_next_hops.max(axis=0) * beta
        #     # 加权属性求和
        #     a_n_h = attr_next_hops.sum(axis=1)
        #     # 求流量划分比
        #     traffic_demand_ratio = a_n_h/a_n_h.sum()
        #
        #     assert np.round(traffic_demand_ratio.sum(), 5) == 1, "Error from the traffic_demand_ratio = %10f" % traffic_demand_ratio.sum()
        #     traffic_demand = traffic_demand_ratio * demand
        #
        # elif policy == 'ECMP':
        #     pass
        #     # traffic_demand = demand / next_hops_cnt * np.ones(next_hops_cnt) # 等分流量
        # else:
        #     pass

        # traffic_demand = [traffic_demand[i] for i in range(len(paths))]
        traffic_demand = demand / len(paths) * np.ones(len(paths))

        for i in range(len(paths)):
            for j in range(len(paths[i]) - 1):
                link_loads[self.link_sd_to_idx[(paths[i][j], paths[i][j + 1])]] += traffic_demand[i]

    def _multipath_next_hop_distribution(self, link_loads, demand, src, dst,
                                         virtual_attr_next_hops, policy='WCMP'):

        if src == dst:
            return

        next_hops = virtual_attr_next_hops[0]
        next_hop_pathnum = virtual_attr_next_hops[1]
        next_hop_hopnum = virtual_attr_next_hops[2]

        next_hops = next_hops[src, dst]  # 例 (0, 7): [10, 13]
        next_hops_cnt = len(next_hops)
        traffic_demand = []

        if policy == 'WCMP':
            attr_next_hops = []  # (1)出口带宽；(2)下一跳后的路径数；(3)下一跳后的不同路径平均跳数（相同下一跳的多个路径的路由跳数平均值）
            beta = np.array([0.5, 0.2, 0.3])  # β1，β2，β3

            for n_p in next_hops:
                attr_n_l_capacity = self.link_capacities[self.link_sd_to_idx[(src, n_p)]]
                attr_n_h_pathnum = next_hop_pathnum[src, dst][n_p]
                attr_n_h_averhopnum = np.array(next_hop_hopnum[src, dst][n_p]).mean()
                attr_next_hops.append([attr_n_l_capacity, 1 / attr_n_h_pathnum, 1 / attr_n_h_averhopnum])
            # 先每个属性除以最大值，归一化 # 再乘以属性影响因子β1，β2，β3
            attr_next_hops = np.array(attr_next_hops)
            attr_next_hops = attr_next_hops / attr_next_hops.max(axis=0) * beta
            # 加权属性求和
            a_n_h = attr_next_hops.sum(axis=1)
            # 求流量划分比
            traffic_demand_ratio = a_n_h / a_n_h.sum()

            assert np.round(traffic_demand_ratio.sum(),
                            5) == 1, "Error from the traffic_demand_ratio = %10f" % traffic_demand_ratio.sum()
            traffic_demand = traffic_demand_ratio * demand

        elif policy == 'ECMP':
            traffic_demand = demand / next_hops_cnt * np.ones(next_hops_cnt)  # 等分流量
        else:
            pass

        traffic_demand = {next_hops[i]: traffic_demand[i] for i in range(len(next_hops))}
        for n_p in next_hops:
            link_loads[self.link_sd_to_idx[(src, n_p)]] += traffic_demand[n_p]
            self._multipath_next_hop_distribution(link_loads, traffic_demand[n_p], n_p, dst,
                                                  virtual_attr_next_hops, policy='WCMP')

    def _get_multipath_next_hops(self, virtual_shortest_paths_node):
        next_hops = {}
        next_hop_pathnum = {}
        next_hop_hopnum = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                next_hops[src, dst] = []
                next_hop_pathnum[src, dst] = {}
                next_hop_hopnum[src, dst] = {}
                for p in virtual_shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in next_hops[src, dst]:
                        next_hops[src, dst].append(p[1])
                        next_hop_pathnum[src, dst][p[1]] = 1
                        next_hop_hopnum[src, dst][p[1]] = [len(p)]
                    else:
                        next_hop_pathnum[src, dst][p[1]] += 1
                        next_hop_hopnum[src, dst][p[1]].append(len(p))

        virtual_attr_next_hops = (next_hops,
                                  next_hop_pathnum,
                                  next_hop_hopnum)
        return virtual_attr_next_hops

    def _crit_links_elect(self, tm_idx, num_crit_links):
        eval_link_loads = self._ecmp_traffic_distribution(tm_idx)
        eval_link_util = eval_link_loads / self.link_capacities

        return np.argsort(eval_link_util)[:num_crit_links]

    def _get_crit_links_flows(self, tm_idx, num_crit_links=10):
        link_loads = self._ecmp_traffic_distribution(tm_idx)
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[
                                :num_crit_links]  # np.argsort函数元素从小到大排列（升序）

        cf_potential = []
        for pair_idx in range(self.num_pairs):
            for path in self.shortest_paths_link[pair_idx]:
                if len(set(path).intersection(critical_link_indexes)) > 0:
                    cf_potential.append(pair_idx)
                    break

        assert len(cf_potential) >= self.max_moves, \
            ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)" % (
                cf_potential, self.max_moves, num_crit_links))

        return critical_link_indexes, self._get_topK_flows(tm_idx, cf_potential)

    def _get_crit_links(self, ecmp_links_util, num_rollout_step=300,
                        offset=150, scale=100, update_interval=1):
        # ##**方法A**#
        #  # update_interval 为关键links更新的时间间隔（即每多少个step更新一次）
        # num_top_link = self.num_crit_links#8#10
        # top_link_indexes = np.argsort(-ecmp_links_util)[:num_top_link]#np.argsort函数元素从小到大排列（升序）
        # self.crit_links_idx_list += top_link_indexes.tolist()
        # if len(self.crit_links_idx_list) > num_rollout_step * num_top_link:
        #     self.crit_links_idx_list = self.crit_links_idx_list[num_top_link:]
        # else:
        #     return self.idx_crit_links # 初始训练时按照默认值
        #
        # if self.step_cnt % update_interval == 0:
        #     assert len(self.crit_links_idx_list) == num_rollout_step*num_top_link, 'ERROR from self._get_crit_links'
        #     arr = np.array(self.crit_links_idx_list)
        #     unique_elements, counts = np.unique(arr, return_counts=True)
        #     idx_counts = np.argsort(-counts)[:self.num_crit_links]
        #     idx_crit_links = [unique_elements[idx_counts[i]] for i in range(self.num_crit_links)]
        #     # idx_crit_links = [32, 60, 28, 62, 19, 52, 41, 22, 20, 39, 51] # 默认值
        #     return idx_crit_links
        # else:
        #     return self.idx_crit_links # 不更新关键links

        ##**方法B**#
        # update_interval 为关键links更新的时间间隔（即每多少个step更新一次）
        num_top_link = self.num_crit_links  #
        top_link_indexes = np.argsort(-ecmp_links_util)[:num_top_link]  # np.argsort函数元素从小到大排列（升序）
        self.crit_links_idx_list += top_link_indexes.tolist()
        if len(self.crit_links_idx_list) > num_rollout_step * num_top_link:
            self.crit_links_idx_list = self.crit_links_idx_list[num_top_link:]

        if self.step_cnt % update_interval == 0:
            # function_score采用gauss函数，x: step, y: score
            N_step = np.arange(num_rollout_step - offset)
            origin = num_rollout_step
            delay = 0.5
            sigma_2 = - scale ** 2 / (2 * np.log(delay))
            S_decay = np.exp(- (np.abs(N_step - origin) - offset) ** 2 / (2 * sigma_2))
            S_offset = np.ones(offset)
            Score = np.round(np.concatenate((S_decay, S_offset)), 4)
            if len(self.crit_links_idx_list) < num_rollout_step * num_top_link:
                Score = Score[-int(len(self.crit_links_idx_list) / num_top_link):]

            # plt.plot(np.arange(num_rollout_step), Score)
            # plt.title('rc.pkl')
            # plt.ylim(0, 1.2)
            # plt.show()

            crit_links_score = np.repeat(Score, num_top_link)

            # 输入数据
            A = np.array(self.crit_links_idx_list)  # link
            B = crit_links_score  # score of link

            # 对A进行排序，同时保持B的顺序与A一致
            idx = np.argsort(A)
            sorted_A = A[idx]
            sorted_B = B[idx]

            # 使用字典统计每个元素的总分
            scores = {}
            for i in range(len(sorted_A)):
                if sorted_A[i] not in scores:
                    scores[sorted_A[i]] = sorted_B[i]
                else:
                    scores[sorted_A[i]] += sorted_B[i]

            # 使用字典解析将值转换为保留四位小数的浮点数
            scores = {k: round(v, 4) for k, v in scores.items()}

            # 输出得分前K的元素及其总分
            top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_K_scores = top_scores[:num_top_link]
            idx_crit_links = [item[0] for item in top_K_scores]
            # print("Top Scores:")
            # for item in top_scores:
            #     print(f"{item[0]}: {item[1]}")

            return idx_crit_links
        else:
            return self.idx_crit_links  # 不更新关键links

    # ###** 对比2：Node-based selection
    def _node_based_selection(self):
        degree = np.zeros(self.num_nodes)
        for i_k in range(self.num_links):
            degree[self.link_idx_to_sd[i_k][0]] += 1  # 计算所有节点的度
        idx_degree_sort = np.argsort(-degree)  # 降序，idx_degree_sort就按照度的大小将node降序排列的结果
        # 从中位数开始往大和往小依次交错排序
        median = int(len(idx_degree_sort) / 2)
        A_list = np.arange(median, self.num_nodes)  # 从中位数 开始往大
        B_list = np.arange(median - 1, -1, -1)  # 从中位数-1 开始往小
        idx_median_list = []
        for i_AB in range(len(B_list)):
            idx_median_list += [A_list[i_AB]] + [B_list[i_AB]]  # 从中位数开始往大和往小依次交错排序成一个list
        if len(A_list) % 2 != 1:
            idx_median_list += [A_list[-1]]  ##中位数时将A_list最后一个补上

        filtered_keys = []
        for i_mk in idx_median_list:
            idx_key = idx_degree_sort[i_mk]  # 将交错排序后的索引置换成idx_degree_sort的索引
            filtered_keys += [self.link_sd_to_idx[k] for k in self.link_sd_to_idx.keys() if k[0] == idx_key]  # 从中位数开始往大和往小依次交错取出并按顺序排列链接到各个节点的links
        idx_crit_links = np.array(filtered_keys)[:self.num_crit_links]  # 取前K个关键links
        return idx_crit_links

    ###** 对比3：Centrality-based selection
    def _centrality_based_selection(self):
        centrality = np.zeros(self.num_links)

        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                path = self.shortest_paths_link[self.pair_sd_to_idx[(src, dst)]][0]  # 最短路径,for循环仅进行一次
                for p_e in path:
                    centrality[p_e] += 1  # 累积计算所有link的中心度

        ###**
        return np.argsort(-centrality)[:self.num_crit_links]
        ###**
        # idx_centrality_sort = np.argsort(-centrality)  # 降序，idx_degree_sort就按照中心度的大小将link降序排列的结果
        # ## 从中位数开始往大和往小依次交错排序
        # median = int(len(idx_centrality_sort) / 2)
        # A_list = np.arange(median, self.num_links)  # 从中位数 开始往大
        # B_list = np.arange(median - 1, -1, -1)  # 从中位数-1 开始往小
        # idx_median_list = []
        # for i_AB in range(len(B_list)):
        #     idx_median_list += [A_list[i_AB]] + [B_list[i_AB]]  # 从中位数开始往大和往小依次交错排序成一个list
        # if len(A_list) % 2 != 1:
        #     idx_median_list += [A_list[-1]]  ##中位数时将A_list最后一个补上
        #
        # interleaving_sort = []
        # for i_mk in idx_median_list:
        #     idx_key = idx_centrality_sort[i_mk]  # 将交错排序后的索引置换成idx_centrality_sort的索引
        #     interleaving_sort.append(idx_key)
        # idx_crit_links = np.array(interleaving_sort)[:self.num_crit_links]  # 取前K个关键links
        # return idx_crit_links

    ############*************
    def _e_c_l_and_c_f_rewrite(self, tm_idx, action_CL):  # action_CL中的值范围[0,1]
        eval_tm = self.traffic_matrices[tm_idx]
        eval_link_loads = np.zeros((self.num_links))
        # 挑选关键流
        self.dict_link_tm_demand = {}  #

        for i in range(self.num_links):  #
            self.dict_link_tm_demand[i] = np.zeros(self.num_pairs)  #

        idx_crit_flows = self._get_topK_flows(self.tm_idx, self.flow_pairs)  # Top K
        # 使用全部的真实link权重所得到的默认多路径路由，对非关键流进行流量路由分配
        for idx_DF in range(self.num_pairs):
            if idx_DF not in idx_crit_flows:
                s, d = self.pair_idx_to_sd[idx_DF]
                demand = eval_tm[s][d]
                if demand != 0:
                    self._e_n_h_d_rewrite(eval_link_loads, demand, s, d, idx_DF)  #

        virtual_shortest_paths_node = []
        virtual_DG = copy.deepcopy(self.DG)
        # 更改link权重信息
        for i in range(len(self.idx_crit_links)):
            idx_CL = self.idx_crit_links[i]
            s, d = self.link_idx_to_sd[idx_CL]
            original_weight = self.DG.edges[s, d]['weight']
            virtual_DG.edges[s, d]['weight'] = original_weight * (self.link_capacities[idx_CL]
                                                                  / (OBJ_EPSILON + self.link_capacities[idx_CL] *
                                                                     action_CL[i]))

        # 使用关键links权重更改后的网络，生成新的多路径路由
        # f = open(shortest_paths_file, 'w+')
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s != d:
                    virtual_shortest_paths_node.append(
                        list(nx.all_shortest_paths(virtual_DG, s, d, weight='weight', method='dijkstra')))
                    # line = str(s) + '->' + str(d) + ': ' + str(virtual_shortest_paths_node[-1])
                    # f.writelines(line + '\n')

        virtual_shortest_paths_link = self._convert_to_edge_path(virtual_shortest_paths_node)
        virtual_attr_next_hops = self._get_multipath_next_hops(virtual_shortest_paths_node)
        # 使用新的虚拟多路径路由，对关键流进行流量路由分配
        for idx_CF in idx_crit_flows:
            s, d = self.pair_idx_to_sd[idx_CF]
            demand = eval_tm[s][d]
            if demand != 0:
                self._m_n_h_d_rewrite(idx_CF, eval_link_loads, demand, s, d,
                                      virtual_attr_next_hops, policy='WCMP')

        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)

        # delay = 0
        # if eval_delay:
        #     assert tm_idx in self.load_multiplier, (tm_idx)
        #     eval_link_loads *= self.load_multiplier[tm_idx]
        #     delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))
        return self.dict_link_tm_demand

    def _m_n_h_d_rewrite(self, idx_CF, link_loads, demand, src, dst,
                         virtual_attr_next_hops, policy='WCMP'):

        if src == dst:
            return

        next_hops = virtual_attr_next_hops[0]
        next_hop_pathnum = virtual_attr_next_hops[1]
        next_hop_hopnum = virtual_attr_next_hops[2]

        next_hops = next_hops[src, dst]  # 例 (0, 7): [10, 13]
        next_hops_cnt = len(next_hops)
        traffic_demand = []

        if policy == 'WCMP':
            attr_next_hops = []  # (1)出口带宽；(2)下一跳后的路径数；(3)下一跳后的不同路径平均跳数（相同下一跳的多个路径的路由跳数平均值）
            beta = np.array([0.5, 0.2, 0.3])  # β1，β2，β3

            for n_p in next_hops:
                attr_n_l_capacity = self.link_capacities[self.link_sd_to_idx[(src, n_p)]]
                attr_n_h_pathnum = next_hop_pathnum[src, dst][n_p]
                attr_n_h_averhopnum = np.array(next_hop_hopnum[src, dst][n_p]).mean()
                attr_next_hops.append([attr_n_l_capacity, 1 / attr_n_h_pathnum, 1 / attr_n_h_averhopnum])
            # 先每个属性除以最大值，归一化 # 再乘以属性影响因子β1，β2，β3
            attr_next_hops = np.array(attr_next_hops)
            attr_next_hops = attr_next_hops / attr_next_hops.max(axis=0) * beta
            # 加权属性求和
            a_n_h = attr_next_hops.sum(axis=1)
            # 求流量划分比
            traffic_demand_ratio = a_n_h / a_n_h.sum()

            assert np.round(traffic_demand_ratio.sum(),
                            5) == 1, "Error from the traffic_demand_ratio = %10f" % traffic_demand_ratio.sum()
            traffic_demand = traffic_demand_ratio * demand

        elif policy == 'ECMP':
            traffic_demand = demand / next_hops_cnt * np.ones(next_hops_cnt)  # 等分流量
        else:
            pass

        traffic_demand = {next_hops[i]: traffic_demand[i] for i in range(len(next_hops))}
        for n_p in next_hops:
            link_idx = self.link_sd_to_idx[(src, n_p)]  #
            self.dict_link_tm_demand[link_idx][idx_CF] += traffic_demand[n_p]

            link_loads[self.link_sd_to_idx[(src, n_p)]] += traffic_demand[n_p]
            self._m_n_h_d_rewrite(idx_CF, link_loads, traffic_demand[n_p], n_p, dst,
                                  virtual_attr_next_hops, policy='WCMP')


    def render(self, mode='No'):
        return None

    def close(self):
        return None


def eval_env_step(self, action):
    logging.info(
        "Step the env once: 'step_cnt' is %d and the current index 'tm_idx' is %d" % (self.step_cnt, self.tm_idx))
    eval_delay = False

    assert len(action) == self.num_crit_links + 0, "Error from action"
    action_CL = (action + 1) / 2  # 缩放到[0, 1]

    # ECMP对比
    if not os.path.exists(self.ecmp_mlu_links_util_save_file):
        if self.step_cnt + 1 < self.tm_cnt:
            ecmp_mlu, ecmp_delay, ecmp_links_util = self._eval_ecmp_traffic_distribution(self.tm_idx,
                                                                                         eval_delay=eval_delay)
            self.ecmp_mlu_save.append(ecmp_mlu)
            self.ecmp_links_util_save.append(ecmp_links_util)

        elif self.step_cnt + 1 == self.tm_cnt:

            ecmp_mlu, ecmp_delay, ecmp_links_util = self._eval_ecmp_traffic_distribution(self.tm_idx,
                                                                                         eval_delay=eval_delay)
            self.ecmp_mlu_save.append(ecmp_mlu)
            self.ecmp_links_util_save.append(ecmp_links_util)

            with open(self.ecmp_mlu_links_util_save_file, 'wb') as handle:
                pickle.dump((self.ecmp_mlu_save, self.ecmp_links_util_save), handle)
            ecmp_mlu = self.ecmp_mlu_save[self.tm_idx]
            ecmp_links_util = self.ecmp_links_util_save[self.tm_idx]
    else:
        ecmp_mlu = self.ecmp_mlu_save[self.tm_idx]
        ecmp_links_util = self.ecmp_links_util_save[self.tm_idx]

    mlu, drl_links_util = self._eval_crit_links_and_crit_flows(self.tm_idx, action_CL)

    # start_time = time.time()
    # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, action_CF)
    # end_time = time.time()
    # running_time = end_time - start_time
    # print("代码'optimal_routing_mlu_critical_pairs'的运行时间：", running_time*1000, "ms")
    # mlu, delay = self._eval_critical_flow_and_ecmp(self.tm_idx, action_CF, solution, eval_delay=eval_delay)

    # crit_topk = self._get_critical_topK_flows(self.tm_idx)
    # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, crit_topk)
    # crit_mlu, crit_delay = self._eval_critical_flow_and_ecmp(self.tm_idx, crit_topk, solution, eval_delay=eval_delay)
    #
    # topk = self._get_topK_flows(self.tm_idx, self.lp_pairs)
    # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, topk)
    # topk_mlu, topk_delay = self._eval_critical_flow_and_ecmp(self.tm_idx, topk, solution, eval_delay=eval_delay)

    # _, solution = self._optimal_routing_mlu(self.tm_idx)
    # optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution, eval_delay=eval_delay)

    if not os.path.exists(self.optimal_mlu_save_file):
        if self.step_cnt + 1 < self.tm_cnt:
            _, solution = self._optimal_routing_mlu(self.tm_idx)
            optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution,
                                                                            eval_delay=True)
            self.optimal_mlu_save.append(optimal_mlu)
            self.optimal_mlu_delay_save.append(optimal_mlu_delay)
        elif self.step_cnt + 1 == self.tm_cnt:

            _, solution = self._optimal_routing_mlu(self.tm_idx)
            optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution,
                                                                            eval_delay=True)
            self.optimal_mlu_save.append(optimal_mlu)
            self.optimal_mlu_delay_save.append(optimal_mlu_delay)

            with open(self.optimal_mlu_save_file, 'wb') as handle:
                pickle.dump((self.optimal_mlu_save, self.optimal_mlu_delay_save), handle)
            optimal_mlu = self.optimal_mlu_save[self.tm_idx]
            optimal_mlu_delay = self.optimal_mlu_delay_save[self.tm_idx]
    else:
        optimal_mlu = self.optimal_mlu_save[self.tm_idx]
        optimal_mlu_delay = self.optimal_mlu_delay_save[self.tm_idx]

    norm_mlu = optimal_mlu / mlu
    line = 'im_idx=' + str(self.tm_idx) + ', norm_mlu=' + str(np.around(norm_mlu, decimals=6)) \
           + ', mlu=' + str(np.around(mlu, decimals=6))

    # norm_crit_mlu = optimal_mlu / crit_mlu
    # line += ', norm_crit_mlu=' + str(np.around(norm_crit_mlu, decimals=6)) + ', crit_mlu=' + str(np.around(crit_mlu, decimals=6))
    #
    # norm_topk_mlu = optimal_mlu / topk_mlu
    # line += ', norm_topk_mlu=' + str(np.around(norm_topk_mlu, decimals=6)) + ', topk_mlu=' + str(np.around(topk_mlu, decimals=6))

    norm_ecmp_mlu = optimal_mlu / ecmp_mlu
    line += ', norm_ecmp_mlu=' + str(np.around(norm_ecmp_mlu, decimals=6)) + ', ecmp_mlu=' + str(
        np.around(ecmp_mlu, decimals=6)) + ', '

    # if eval_delay:
    #     solution = self._optimal_routing_delay(self.tm_idx)
    #     optimal_delay = self._eval_optimal_routing_delay(self.tm_idx, solution)
    #
    #     line += str(optimal_delay / delay) + ', '
    #     line += str(optimal_delay / crit_delay) + ', '
    #     line += str(optimal_delay / topk_delay) + ', '
    #     line += str(optimal_delay / optimal_mlu_delay) + ', '
    #     if ecmp:
    #         line += str(optimal_delay / ecmp_delay) + ', '
    #
    #     assert self.tm_idx in self.load_multiplier, (self.tm_idx)
    #     line += str(self.load_multiplier[self.tm_idx]) + ', '
    line = "step_cnt=" + str(self.step_cnt + 1) + ', ' + line
    print(line[:-2])

    # *********计算 reward **********
    reward = (norm_mlu - 0.4) * self.reward_scaling
    print('reward=%0.6f' % reward)

    self.reward_list.append(reward)
    self.r_mlu_list.append(mlu)
    self.optimal_mlu_list.append(optimal_mlu)

    self.norm_mlu_eval_list.append(norm_mlu)  #
    self.norm_ecmp_mlu_eval_list.append(norm_ecmp_mlu)  #

    # if self.step_cnt + 1 == self.total_steps:
    #     with open(self.path_reward_save + '/' + str(self.step_cnt) + '_reward_main_XX.pkl', 'wb') as handle:
    #         pickle.dump((self.reward_list, self.r_mlu_list, self.optimal_mlu_list), handle)

    # *********更新 state **********
    self.tm_idx = self.tm_idx + 1 if self.tm_idx + 1 < self.tm_cnt else 0

    # self.idx_crit_links, self.idx_crit_flows = self._get_crit_links_flows(tm_idx=self.tm_idx, num_crit_links=self.num_crit_links)
    # self.idx_crit_links = [62, 60, 32, 28, 52, 41, 22, 19, 51, 20, 39]#[32, 60, 28, 62, 19, 52, 41, 22, 20, 39, 51]
    # self.idx_crit_links = self._get_crit_links(ecmp_links_util=ecmp_links_util,
    #                                            num_rollout_step=int(self.tm_cnt * 0.6),# !!! 为1时即可得到默认的关键links
    #                                            update_interval=1)

    # num_rollout_step = int(self.tm_cnt * 0.8)
    # self.idx_crit_links = self._get_crit_links(ecmp_links_util=ecmp_links_util,
    #                                            num_rollout_step=num_rollout_step,  # !!! 为1时即可得到默认的关键links
    #                                            offset=int(num_rollout_step * 0.6),
    #                                            scale=int(num_rollout_step * 0.2),
    #                                            update_interval=1)##**scaledrl

    assert len(self.idx_crit_links) == self.num_crit_links, "self.idx_crit_links is not equal to self.num_crit_links"

    # 归一化后的 当前时间步的流量矩阵TM_t， GEANT 维度：(1, 23, 23)
    TM_t = np.array(self.normalized_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
    # 当前时间步的流量矩阵TM_t与上一时间步的流量矩阵TM_(t-1)之差 DTM_t = TM_t - TM_(t-1)，再， GEANT 维度：(1, 23, 23)
    # DTM_t = np.array(self.diff_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
    # # 邻接矩阵AM， GEANT 维度：(1, 23, 23)
    # # AM = nx.to_numpy_array(self.DG, weight=None, nodelist=np.arange(0, self.num_nodes))[np.newaxis, :]
    # # # 关键links信息矩阵 CLIM, GEANT 维度：(1, 23, 23)
    # CLIM = np.zeros((self.num_nodes, self.num_nodes))[np.newaxis, :]
    # for i in self.idx_crit_links:
    #     s, d = self.link_idx_to_sd[i]
    #     CLIM[0][s, d] = 1

    # self.state GEANT 维度：(4, 23, 23)
    self.state = TM_t  # np.concatenate((TM_t, DTM_t, CLIM), axis=0)  # 拼接##**scaledrl
    # 可以考虑对所有输入网络的state用固定的均值和方差进行归一化

    # done = True if (self.step_cnt+1) % int(self.num_steps_per_episode) == 0 else False
    done = False
    info = {}

    self.step_cnt += 1

    return self.state, reward, done, info
    ## self.state为array类型('numpy.ndarray')，reward为float类型，done为False(episode未终止)或True（episode终止）

################################################################
##############*** 下面代码为区分train与test ***####################
################################################################


# def env_init_eval(self): # Abilene，GEANT
#     logging.info("Create and initialize the env")
#     # self.total_steps = None
#     #
#     config = get_config(FLAGS) or FLAGS
#     # self.num_steps_per_episode = config.num_steps_per_episode
#     # self.data_dir = config.data_root + '/'
#     # self.path_reward_save = path_reward_save
#     is_training = False
#     if config.topology_file == 'Abilene':  # *#
#         # self.topology = AbileneTopology(config, self.data_dir)
#         self.traffic = AbileneTraffic(config, self.topology.num_nodes, self.data_dir, is_training=is_training)
#     elif config.topology_file == 'GEANT':  # *#
#         # self.topology = GEANTTopology(config, self.data_dir)
#         self.traffic = GEANTTraffic(config, self.topology, self.data_dir, is_training=is_training)
#    # 应特别主要 eval下的self.tm_cnt已经改变
#     elif config.topology_file == 'Xspedius':  # *#
#         # self.topology = utlis.network_env.XspediusTopology(config, self.data_dir)
#         self.traffic = utlis.network_env.XspediusTraffic(config, self.topology, self.data_dir,
#                                                          is_training=is_training)
#     elif config.topology_file == 'SwitchL3':  # *#
#         # self.topology = utlis.network_env.SwitchL3Topology(config, self.data_dir)
#         self.traffic = utlis.network_env.SwitchL3Traffic(config, self.topology, self.data_dir,
#                                                          is_training=is_training)
#     elif config.topology_file == 'Uunet':  # *#
#         # self.topology = utlis.network_env.UunetTopology(config, self.data_dir)
#         self.traffic = utlis.network_env.UunetTraffic(config, self.topology, self.data_dir,
#                                                       is_training=is_training)
#     elif config.topology_file == 'Dfn':
#         # self.topology = utlis.network_env.DfnTopology(config, self.data_dir)
#         self.traffic = utlis.network_env.DfnTraffic(config, self.topology, self.data_dir,
#                                                     is_training=is_training)
#     else:
#         pass
#
#     self.topology_file = config.topology_file
#     if self.topology_file == 'Abilene':
#         self.traffic_matrices = self.traffic.traffic_matrices * 100 * 8 / 300 / 1000  # 转为单位kbps ## Unit: (100 bytes / 5 minutes)  // 100 is the pkt sampling rate
#     elif self.topology_file == 'GEANT':
#         self.traffic_matrices = self.traffic.traffic_matrices  # 原始数据集单位kbps，<unit type="bandwidth" value="kbps"/>
#     elif self.topology_file == 'Xspedius':
#         self.traffic_matrices = self.traffic.traffic_matrices
#     elif self.topology_file == 'SwitchL3':
#         self.traffic_matrices = self.traffic.traffic_matrices
#     elif self.topology_file == 'Uunet':
#         self.traffic_matrices = self.traffic.traffic_matrices
#     else:
#         pass
#
#     self.traffic_matrices_dims = self.traffic_matrices.shape
#     self.tm_cnt = self.traffic.tm_cnt
#     # self.tm_cnt = self.tm_cnt #**##
#     print('self.tm_cnt = %d'%self.tm_cnt)#
#     self.traffic_file = self.traffic.traffic_file
#     # self.num_pairs = self.topology.num_pairs
#     # self.pair_idx_to_sd = self.topology.pair_idx_to_sd
#     # self.pair_sd_to_idx = self.topology.pair_sd_to_idx
#     # self.num_nodes = self.topology.num_nodes
#     # self.num_links = self.topology.num_links
#     # self.link_idx_to_sd = self.topology.link_idx_to_sd
#     # self.link_sd_to_idx = self.topology.link_sd_to_idx
#     # self.link_capacities = self.topology.link_capacities  # capacity(kbps)
#     # self.link_weights = self.topology.link_weights
#     # self.shortest_paths_node = self.topology.shortest_paths  # paths consist of nodes
#     # self.shortest_paths_link = self._convert_to_edge_path(self.shortest_paths_node)
#     # self.DG = self.topology.DG
#
#     # self._get_ecmp_next_hops()
#     #
#     # self.flow_pairs = [p for p in range(self.num_pairs)]
#     # # for LP
#     # self.lp_pairs = [p for p in range(self.num_pairs)]
#     # self.lp_nodes = [n for n in range(self.num_nodes)]
#     # self.links = [e for e in range(self.num_links)]
#     # self.lp_links = [e for e in self.link_sd_to_idx]
#     # self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]
#     #
#     # self.load_multiplier = {}
#
#     # self.project_name = config.project_name
#     # self.num_crit_links = int(np.round(self.num_links * 0.15))  # math.ceil(0.15 * self.num_links)  # 10%*N
#     # self.max_moves = math.ceil(0.15 * self.num_pairs)  # TopK流的K值#10%*N
#
#     self.tm_indexes = np.arange(0, self.tm_cnt)
#     self.valid_tm_cnt = len(self.tm_indexes)
#
#     self.ecmp_mlu_links_util_save_file_eval = self.data_dir + self.topology_file + '/' + self.topology_file + '_ecmp_mlu_links_util_save_eval.pkl'
#     if os.path.exists(self.ecmp_mlu_links_util_save_file_eval):
#         with open(self.ecmp_mlu_links_util_save_file_eval, 'rb') as pkl:
#             self.ecmp_mlu_save, self.ecmp_links_util_save = pickle.load(pkl)
#         assert len(self.ecmp_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
#     else:
#         self.ecmp_mlu_save = []
#         self.ecmp_links_util_save = []
#
#     self.optimal_mlu_save_file_eval = self.data_dir + self.topology_file + '/' + self.topology_file + '_optimal_mlu_save_eval.pkl'
#     if os.path.exists(self.optimal_mlu_save_file_eval):
#         with open(self.optimal_mlu_save_file_eval, 'rb') as pkl:
#             self.optimal_mlu_save, self.optimal_mlu_delay_save = pickle.load(pkl)
#         assert len(self.optimal_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
#     else:
#         self.optimal_mlu_save = []
#         self.optimal_mlu_delay_save = []
#
#     self._generate_inputs(norm_type='MaxMin')#!要保留
#     self.state_dims = self.normalized_traffic_matrices.shape[1:]
#     print('Input dims :', self.state_dims)
#     print('Number of critical links :', self.num_crit_links)
#
#     # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_crit_links + 0,), dtype=np.float32)
#     # self.observation_space = spaces.Box(low=0, high=1.0, shape=(3, self.num_nodes, self.num_nodes), dtype=np.float32)
#
#  #***#以下是reset的阶段##
#     self.tm_idx = 0
#     #**# self.idx_crit_links 不能采用随机初始化，应该紧接training的最后一个step的..
#     # self.idx_crit_links = np.random.randint(0, self.num_crit_links, size=self.num_crit_links)  # 随机初始化关键链路
#     assert len(self.idx_crit_links) == self.num_crit_links, "self.idx_crit_links is not equal to self.num_crit_links"
#
#     TM_t = np.array(self.normalized_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
#     DTM_t = np.array(self.diff_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
#     CLIM = np.zeros((self.num_nodes, self.num_nodes))[np.newaxis, :]
#     for i in self.idx_crit_links:
#         s, d = self.link_idx_to_sd[i]
#         CLIM[0][s, d] = 1
#     self.state = np.concatenate((TM_t, DTM_t, CLIM), axis=0)  # 拼接
#
#     # self.reward_scaling = 1 / 0.6
#     self.reward_list = []
#     self.r_mlu_list = []
#     self.optimal_mlu_list = []
#     self.optimal_mlu_delay_list = []
#     self.crit_links_idx_list = []
#     self.crit_links_idx_bidir_list = []
#
#     self.step_cnt = 0
#     # return self.state
#
#     return self#考虑注释掉
#
# # def env_init_simu_TM_eval(self): # 其他
# #
# #     return self
#
# def eval_env_step(self, action):
#     logging.info(
#         "Step the env once: 'step_cnt' is %d and the current index 'tm_idx' is %d" % (self.step_cnt, self.tm_idx))
#     eval_delay = False
#
#     assert len(action) == self.num_crit_links + 0, "Error from action"
#     action_CL = (action + 1) / 2  # 缩放到[0, 1]
#
#     # ECMP对比
#     if not os.path.exists(self.ecmp_mlu_links_util_save_file_eval):
#         if self.step_cnt + 1 < self.tm_cnt:
#             ecmp_mlu, ecmp_delay, ecmp_links_util = self._eval_ecmp_traffic_distribution(self.tm_idx,
#                                                                                          eval_delay=eval_delay)
#             self.ecmp_mlu_save.append(ecmp_mlu)
#             self.ecmp_links_util_save.append(ecmp_links_util)
#
#         elif self.step_cnt + 1 == self.tm_cnt:
#
#             ecmp_mlu, ecmp_delay, ecmp_links_util = self._eval_ecmp_traffic_distribution(self.tm_idx,
#                                                                                          eval_delay=eval_delay)
#             self.ecmp_mlu_save.append(ecmp_mlu)
#             self.ecmp_links_util_save.append(ecmp_links_util)
#
#             with open(self.ecmp_mlu_links_util_save_file_eval, 'wb') as handle:
#                 pickle.dump((self.ecmp_mlu_save, self.ecmp_links_util_save), handle)
#             ecmp_mlu = self.ecmp_mlu_save[self.tm_idx]
#             ecmp_links_util = self.ecmp_links_util_save[self.tm_idx]
#     else:
#         ecmp_mlu = self.ecmp_mlu_save[self.tm_idx]
#         ecmp_links_util = self.ecmp_links_util_save[self.tm_idx]
#
#     mlu, drl_links_util = self._eval_crit_links_and_crit_flows(self.tm_idx, action_CL)
#
#     # start_time = time.time()
#     # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, action_CF)
#     # end_time = time.time()
#     # running_time = end_time - start_time
#     # print("代码'optimal_routing_mlu_critical_pairs'的运行时间：", running_time*1000, "ms")
#     # mlu, delay = self._eval_critical_flow_and_ecmp(self.tm_idx, action_CF, solution, eval_delay=eval_delay)
#
#     # crit_topk = self._get_critical_topK_flows(self.tm_idx)
#     # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, crit_topk)
#     # crit_mlu, crit_delay = self._eval_critical_flow_and_ecmp(self.tm_idx, crit_topk, solution, eval_delay=eval_delay)
#     #
#     # topk = self._get_topK_flows(self.tm_idx, self.lp_pairs)
#     # _, solution = self._optimal_routing_mlu_critical_pairs(self.tm_idx, topk)
#     # topk_mlu, topk_delay = self._eval_critical_flow_and_ecmp(self.tm_idx, topk, solution, eval_delay=eval_delay)
#
#     # _, solution = self._optimal_routing_mlu(self.tm_idx)
#     # optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution, eval_delay=eval_delay)
#
#     if not os.path.exists(self.optimal_mlu_save_file_eval):
#         if self.step_cnt + 1 < self.tm_cnt:
#             _, solution = self._optimal_routing_mlu(self.tm_idx)
#             optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution,
#                                                                             eval_delay=True)
#             self.optimal_mlu_save.append(optimal_mlu)
#             self.optimal_mlu_delay_save.append(optimal_mlu_delay)
#         elif self.step_cnt + 1 == self.tm_cnt:
#
#             _, solution = self._optimal_routing_mlu(self.tm_idx)
#             optimal_mlu, optimal_mlu_delay = self._eval_optimal_routing_mlu(self.tm_idx, solution,
#                                                                             eval_delay=True)
#             self.optimal_mlu_save.append(optimal_mlu)
#             self.optimal_mlu_delay_save.append(optimal_mlu_delay)
#
#             with open(self.optimal_mlu_save_file_eval, 'wb') as handle:
#                 pickle.dump((self.optimal_mlu_save, self.optimal_mlu_delay_save), handle)
#             optimal_mlu = self.optimal_mlu_save[self.tm_idx]
#             optimal_mlu_delay = self.optimal_mlu_delay_save[self.tm_idx]
#     else:
#         optimal_mlu = self.optimal_mlu_save[self.tm_idx]
#         optimal_mlu_delay = self.optimal_mlu_delay_save[self.tm_idx]
#
#     norm_mlu = optimal_mlu / mlu
#     line = 'im_idx=' + str(self.tm_idx) + ', norm_mlu=' + str(np.around(norm_mlu, decimals=6)) \
#            + ', mlu=' + str(np.around(mlu, decimals=6))
#
#     # norm_crit_mlu = optimal_mlu / crit_mlu
#     # line += ', norm_crit_mlu=' + str(np.around(norm_crit_mlu, decimals=6)) + ', crit_mlu=' + str(np.around(crit_mlu, decimals=6))
#     #
#     # norm_topk_mlu = optimal_mlu / topk_mlu
#     # line += ', norm_topk_mlu=' + str(np.around(norm_topk_mlu, decimals=6)) + ', topk_mlu=' + str(np.around(topk_mlu, decimals=6))
#
#     norm_ecmp_mlu = optimal_mlu / ecmp_mlu
#     line += ', norm_ecmp_mlu=' + str(np.around(norm_ecmp_mlu, decimals=6)) + ', ecmp_mlu=' + str(
#         np.around(ecmp_mlu, decimals=6)) + ', '
#
#     # if eval_delay:
#     #     solution = self._optimal_routing_delay(self.tm_idx)
#     #     optimal_delay = self._eval_optimal_routing_delay(self.tm_idx, solution)
#     #
#     #     line += str(optimal_delay / delay) + ', '
#     #     line += str(optimal_delay / crit_delay) + ', '
#     #     line += str(optimal_delay / topk_delay) + ', '
#     #     line += str(optimal_delay / optimal_mlu_delay) + ', '
#     #     if ecmp:
#     #         line += str(optimal_delay / ecmp_delay) + ', '
#     #
#     #     assert self.tm_idx in self.load_multiplier, (self.tm_idx)
#     #     line += str(self.load_multiplier[self.tm_idx]) + ', '
#     line = "step_cnt=" + str(self.step_cnt + 1) + ', ' + line
#     print(line[:-2])
#
#     # *********计算 reward **********
#     reward = (norm_mlu - 0.4) * self.reward_scaling
#     print('reward=%0.6f' % reward)
#
#     self.reward_list.append(reward)
#     self.r_mlu_list.append(mlu)
#     self.optimal_mlu_list.append(optimal_mlu)
#
#     self.norm_mlu_eval_list.append(norm_mlu)#
#     self.norm_ecmp_mlu_eval_list.append(norm_ecmp_mlu)#
#     # if self.step_cnt + 1 == self.total_steps:
#     #     with open(self.path_reward_save + '/' + str(self.step_cnt) + '_reward_main_XX.pkl', 'wb') as handle:
#     #         pickle.dump((self.reward_list, self.r_mlu_list, self.optimal_mlu_list), handle)
#
#     # *********更新 state **********
#     self.tm_idx = self.tm_idx + 1 if self.tm_idx + 1 < self.tm_cnt else 0
#
#     # self.idx_crit_links, self.idx_crit_flows = self._get_crit_links_flows(tm_idx=self.tm_idx, num_crit_links=self.num_crit_links)
#     # self.idx_crit_links = [62, 60, 32, 28, 52, 41, 22, 19, 51, 20, 39]#[32, 60, 28, 62, 19, 52, 41, 22, 20, 39, 51]
#     # self.idx_crit_links = self._get_crit_links(ecmp_links_util=ecmp_links_util,
#     #                                            num_rollout_step=int(self.tm_cnt * 0.6),# !!! 为1时即可得到默认的关键links
#     #                                            update_interval=1)
#     num_rollout_step = int(self.tm_cnt * 0.8)
#     self.idx_crit_links = self._get_crit_links(ecmp_links_util=ecmp_links_util,
#                                                num_rollout_step=num_rollout_step,  # !!! 为1时即可得到默认的关键links
#                                                offset=int(num_rollout_step * 0.6),
#                                                scale=int(num_rollout_step * 0.2),
#                                                update_interval=1)
#
#     assert len(self.idx_crit_links) == self.num_crit_links, "self.idx_crit_links is not equal to self.num_crit_links"
#
#     # 归一化后的 当前时间步的流量矩阵TM_t， GEANT 维度：(1, 23, 23)
#     TM_t = np.array(self.normalized_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
#     # 当前时间步的流量矩阵TM_t与上一时间步的流量矩阵TM_(t-1)之差 DTM_t = TM_t - TM_(t-1)，再， GEANT 维度：(1, 23, 23)
#     DTM_t = np.array(self.diff_traffic_matrices[self.tm_idx], dtype=np.float32)[np.newaxis, :]
#     # 邻接矩阵AM， GEANT 维度：(1, 23, 23)
#     # AM = nx.to_numpy_array(self.DG, weight=None, nodelist=np.arange(0, self.num_nodes))[np.newaxis, :]
#     # # 关键links信息矩阵 CLIM, GEANT 维度：(1, 23, 23)
#     CLIM = np.zeros((self.num_nodes, self.num_nodes))[np.newaxis, :]
#     # self.idx_crit_links = self._crit_links_elect(tm_idx=self.tm_idx, num_crit_links=self.num_crit_links)
#     for i in self.idx_crit_links:
#         s, d = self.link_idx_to_sd[i]
#         CLIM[0][s, d] = 1
#
#     # self.state GEANT 维度：(4, 23, 23)
#     self.state = np.concatenate((TM_t, DTM_t, CLIM), axis=0)  # 拼接
#     # 可以考虑对所有输入网络的state用固定的均值和方差进行归一化
#
#     # done = True if (self.step_cnt+1) % int(self.num_steps_per_episode) == 0 else False
#     done = False
#     info = {}
#
#     self.step_cnt += 1
#
#     return self.state, reward, done, info
#     ## self.state为array类型('numpy.ndarray')，reward为float类型，done为False(episode未终止)或True（episode终止）



