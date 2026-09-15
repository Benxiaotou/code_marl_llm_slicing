import os
import time
import copy
import pickle
import logging
import random
import math
import pandas as pd
# import copy
import argparse
from argparse import Namespace
import numpy as np
from copy import deepcopy
import networkx as nx
from gym.spaces import Box
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, RawMultiAgentEnv, REGISTRY_MULTI_AGENT_ENV, DummyVecMultiAgentEnv
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import IPPO_Agents, MAPPO_Agents
from operator import itemgetter

import matplotlib.pyplot as plt

import utlis.network_env
from utlis.network_env import AbileneTopology, AbileneTraffic, GEANTTopology, GEANTTraffic
from utlis.general_function import setlogger, setup_seed

OBJ_EPSILON = 1e-12


class CustomMultiAgentEnv(RawMultiAgentEnv):

    def __init__(self, config):
        super(CustomMultiAgentEnv, self).__init__()
        logging.info("Init the env once.")

        self.env_id = config.env_id
        self.data_dir = config.data_root + '/'
        self.path_reward_save = config.path_reward_save
        # self.path_model_save = config.path_model_save
        self.path_env_save = config.path_env_save
        self.save_attach_word = config.save_attach_word

        # Physical Network Topology
        self._init_p_net_topology_traffic(config)

        # 设置基础物理网络的节点和链路属性
        for i in range(self.num_nodes):
            self.DG.nodes[i]['comp_capacity'] = 900 # ！！
            self.DG.nodes[i]['cach_capacity'] = 450 # ！！
            self.DG.nodes[i]['allocated_comp'] = 0  # 物理网络节点中已分配的计算资源（注意”已分配“是指该资源已经分给虚拟业务网络）
            self.DG.nodes[i]['allocated_cach'] = 0  # 物理网络节点中已分配的存储资源

        for i in range(self.num_links):
            s, d = self.link_idx_to_sd[i]
            self.DG[s][d]['bandwidth'] = 30 # ！！ # Gbps
            self.DG[s][d]['allocated_bandw'] = 0  # 物理网络链路中已分配的带宽资源
            self.link_capacities = 30 # ！！ # Gbps #重新修改初始的链路容量参数
        DG_save = deepcopy(self.DG)

        # 查看拓扑属性信息
        # self.DG_p_net.nodes(data=True)
        # self.DG_p_net.edges(data=True)

        ###############

        # # 设置已有虚拟业务网络 k1、k2、k3...
        # self.K_num_existing_v_net = 3  # 论文中的K
        # self.k1_DG = deepcopy(self.DG)
        # self.k2_DG = deepcopy(self.DG)
        # self.k3_DG = deepcopy(self.DG)
        #
        # # 初始化并设置已有虚拟业务网络的节点和链路属性
        # str_v_net_name = 'k1'
        # self.k1_net = InitVnetwork(config, self.data_dir, self.k1_DG, str_v_net_name)
        # self.k1_net = self._set_v_network_attr(v_net=self.k1_net, comp_capacity=300, cach_capacity=150, bandwidth=10)
        #
        # str_v_net_name = 'k2'
        # self.k2_net = InitVnetwork(config, self.data_dir, self.k2_DG, str_v_net_name)
        # self.k2_net = self._set_v_network_attr(v_net=self.k2_net, comp_capacity=300, cach_capacity=150, bandwidth=10)
        #
        # str_v_net_name = 'k3'
        # self.k3_net = InitVnetwork(config, self.data_dir, self.k3_DG, str_v_net_name)
        # self.k3_net = self._set_v_network_attr(v_net=self.k3_net, comp_capacity=300, cach_capacity=150, bandwidth=10)
        #
        # # 设置新生业务网络
        # self.g_DG = deepcopy(self.DG)
        # str_v_net_name = 'g'
        # self.g_net = InitVnetwork(config, self.data_dir, self.g_DG, str_v_net_name)
        # self.g_net = self._set_g_network_attr(g_net=self.g_net, comp_capacity=300, cach_capacity=150, bandwidth=10)  # !!

        # observations

        ################ 构造包含k_net和g_net数据集
        remove_nodes_set = [[2, 3, 4, 7, 8, 11, 14, 16, 17, 19, 22],  # num_links = 34
                            [0, 3, 4, 5, 6, 8, 11, 16, 17, 19, 22],  # num_links = 36
                            [1, 2, 3, 4, 8, 11, 14, 16, 17, 19, 22],  # num_links = 36
                            [0, 2, 3, 5, 6, 7, 11, 13, 14, 16, 18],  # num_links = 28
                            [0, 2, 3, 4, 5, 8, 11, 12, 14, 16, 21],  # num_links = 30
                            [0, 2, 3, 4, 5, 6, 8, 11, 13, 14, 18],  # num_links = 32
                            [1, 2, 3, 4, 5, 7, 8, 11, 16, 19, 22],  # num_links = 30
                            [0, 1, 2, 3, 4, 5, 8, 10, 11, 14, 16],  # num_links = 28
                            [0, 1, 4, 9, 10, 11, 16, 17, 18, 19, 22],  # num_links = 30
                            [0, 3, 4, 8, 11, 14, 16, 17, 18, 19, 22]]  # num_links = 36

        re_n_set_num_nodes = 12  # ！移出的节点数量
        re_n_set_num_links = [34, 36, 36, 28, 30, 32, 30, 28, 30, 36]  # ！各个net移出的链路数量
        num_re_k = 6  # 前 num_re_k 个用于k_net
        k_remove_nodes_set = remove_nodes_set[0:num_re_k]  # 前6个用于k_net
        g_remove_nodes_set = remove_nodes_set[num_re_k:]  # 后4个用于g_net

        self.K_num_existing_v_net = 3  # 论文中的 K
        self.sample_cnt = 600  # 样本数量

        self.sample_dict_save_file = self.data_dir + self.topology_file + '/' \
                                     + self.topology_file + '_' + str(self.K_num_existing_v_net) \
                                     + '_' + str(re_n_set_num_nodes) + '_' + str(self.sample_cnt) + '_sample_save.pkl'
        # 拓扑名 + k_net个数 + 移除的节点数 + 样本数量 + '_sample_save.pkl'
        if os.path.exists(self.sample_dict_save_file):
            with open(self.sample_dict_save_file, 'rb') as pkl:
                self.sample_dict = pickle.load(pkl)
            assert self.sample_cnt == len(self.sample_dict), "Error from the self.sample_dict_save_file"
            # if isinstance(self.sample_dict[0], tuple):
            #     pass
            # else:
            #     for i_s in range(self.sample_cnt):
            #         self.sample_dict[i_s] = tuple(self.sample_dict[i_s])

        else:
            self.sample_dict = {}  # 每个样本为 [k1, k2, k3,.., g, self.DG]
            for i_s in range(self.sample_cnt):
                del self.DG
                self.DG = deepcopy(DG_save)  # 重新取出 self.DG
                logging.info(str(i_s))  #
                self.sample_dict[i_s] = []
                ## k_net
                re_idx = np.random.choice(np.arange(len(k_remove_nodes_set)), self.K_num_existing_v_net).tolist()
                # comp_capacity = np.random.uniform(230, 330, (3,)).tolist()
                for i_k in range(self.K_num_existing_v_net):
                    str_v_net_name = 'v'
                    v_DG = deepcopy(DG_save)
                    v_remove_nodes_idx = k_remove_nodes_set[re_idx[i_k]]
                    k_net = InitVnetworkTrain(config, self.data_dir, v_DG, str_v_net_name, v_remove_nodes_idx)
                    comp_capacity = 300 * np.clip(np.random.normal(loc=1, scale=0.1), 0.9, 1.1)
                    # np.random.uniform(230, 330)
                    cach_capacity = 150 * np.clip(np.random.normal(loc=1, scale=0.1), 0.9, 1.1)
                    # np.random.uniform(230, 330)
                    bandwidth = 10 * np.clip(np.random.normal(loc=1, scale=0.1), 0.9, 1.1)
                    # np.random.uniform(115, 165)
                    k_net = self._set_v_network_attr(v_net=k_net,
                                                     comp_capacity=comp_capacity,
                                                     cach_capacity=cach_capacity,
                                                     bandwidth=bandwidth)  # 300, 150 ,10
                    print("comp_capacity = " + str(comp_capacity) + "; cach_capacity = " + str(
                        cach_capacity) + "; bandwidth = " + str(bandwidth))
                    self.sample_dict[i_s] += [k_net]
                ## g_net
                re_idx = np.random.choice(np.arange(len(g_remove_nodes_set)), 1).item()
                str_v_net_name = 'v'
                v_DG = deepcopy(self.DG)
                v_remove_nodes_idx = g_remove_nodes_set[re_idx]
                g_net = InitVnetworkTrain(config, self.data_dir, v_DG, str_v_net_name, v_remove_nodes_idx)
                comp_capacity = 300 * np.clip(np.random.normal(loc=1, scale=0.1), 0.9,
                                              1.1)  # np.random.uniform(230, 330)
                cach_capacity = 150 * np.clip(np.random.normal(loc=1, scale=0.1), 0.9,
                                              1.1)  # np.random.uniform(230, 330)
                bandwidth = 10 * np.clip(np.random.normal(loc=1, scale=0.1), 0.9, 1.1)  # np.random.uniform(115, 165)
                g_net = self._set_g_network_attr(g_net=g_net,
                                                 comp_capacity=comp_capacity,
                                                 cach_capacity=cach_capacity,
                                                 bandwidth=bandwidth)  # 300, 150 ,10
                self.sample_dict[i_s] += [g_net]
                self.sample_dict[i_s] += [self.DG]  # 注意 self.DG同样需要保存
                # 转为元组形式
                self.sample_dict[i_s] = tuple(self.sample_dict[i_s])

                # 检查 allocated_bandw
                for i in range(self.num_links):
                    s, d = self.link_idx_to_sd[i]
                    if self.DG[s][d]['allocated_bandw'] > self.DG[s][d]['bandwidth']:
                        print("Error ")
                    assert self.DG[s][d]['allocated_bandw'] <= self.DG[s][d]['bandwidth'], \
                        "Error from the allocated_bandw."
                # # 检查
                # allocated_bandw_list = []
                # bandwidth_list = []
                # for idx_link in range(self.num_links):
                #     s, d = self.link_idx_to_sd[idx_link]
                #     bandwidth_list.append(self.DG[s][d]['bandwidth'])  # Gbps
                #     allocated_bandw_list.append(self.DG[s][d]['allocated_bandw'])
                # if max(allocated_bandw_list) > 33:
                #     print("Error")

            ## 保存数据集
            with open(self.sample_dict_save_file, 'wb') as handle:
                pickle.dump(self.sample_dict, handle)

        # # 检查
        # for i_s in range(self.sample_cnt):
        #     self.k1_net, self.k2_net, self.k3_net, self.g_net, self.DG = self.sample_dict[i_s]
        #     allocated_bandw_list = []
        #     bandwidth_list = []
        #     for idx_link in range(self.num_links):
        #         s, d = self.link_idx_to_sd[idx_link]
        #         bandwidth_list.append(self.DG[s][d]['bandwidth'])  # Gbps
        #         allocated_bandw_list.append(self.DG[s][d]['allocated_bandw'])
        #     if max(allocated_bandw_list) > 33:
        #         print("Error")

        self.num_agents = 1 + self.K_num_existing_v_net  # agent总数量
        self.possible_agents = ["agent_k" + str(r + 1) for r in range(self.K_num_existing_v_net)] + ["agent_g"]
        self.agents = self.possible_agents[:]

        self.max_num_links = max(re_n_set_num_links)
        obs_k_one_shape = 5 * re_n_set_num_nodes + 3 * self.max_num_links + 2 * self.max_num_links  # 节点信息(计算)，链路信息，拓扑信息
        # ### !!特别注意上句代码错误实际应该更改为：
        # obs_k_one_shape = 5 * (self.DG.number_of_nodes() - re_n_set_num_nodes) \
        #                   + 3 * self.max_num_links + 2 * self.max_num_links
        # ### !!
        # obs_g_one_shape = 4 * re_n_set_num_nodes + 2 * self.max_num_links + 2 * self.max_num_links  # 节点信息(计算)，链路信息，拓扑信息
        obs_g_one_shape = obs_k_one_shape  # 补零
        self.state_space = Box(low=-1.0, high=1.0,
                               shape=(self.K_num_existing_v_net * obs_k_one_shape + obs_g_one_shape,), dtype=np.float32)
        self.observation_space = {self.agents[0]: Box(low=0, high=1.0, shape=[obs_k_one_shape, ], dtype=np.float32),
                                  self.agents[1]: Box(low=0, high=1.0, shape=[obs_k_one_shape, ], dtype=np.float32),
                                  self.agents[2]: Box(low=0, high=1.0, shape=[obs_k_one_shape, ], dtype=np.float32),
                                  self.agents[-1]: Box(low=0, high=1.0, shape=[obs_g_one_shape, ], dtype=np.float32)}
        # 注意：obs_k_one_shape应该与obs_k_one_shape相同，即：每个智能体的观测空间大小是相同的
        # 因此需要将obs_g_one_shape补零到与obs_k_one_shape相同，
        # 补零方案1：每个节点、链路均补一个0；方案2：在obs_g后面补上多个零

        self.action_k_one_shape = 4  # 粗粒度控制节点和链路资源的调整比例
        self.action_k_node_control = 2  # 粗粒度控制节点资源的调整比例
        self.action_k_link_control = 2  # 粗粒度控制链路资源的调整比例
        self.action_g_one_shape = self.action_k_one_shape  # 3  # 分别控制计算、缓存、带宽资源的调整比例
        self.action_space = {self.agents[0]: Box(low=-1, high=1.0, shape=[self.action_k_one_shape, ], dtype=np.float32),
                             self.agents[1]: Box(low=-1, high=1.0, shape=[self.action_k_one_shape, ], dtype=np.float32),
                             self.agents[2]: Box(low=-1, high=1.0, shape=[self.action_k_one_shape, ], dtype=np.float32),
                             self.agents[-1]: Box(low=-1, high=1.0, shape=[self.action_g_one_shape, ],
                                                  dtype=np.float32)}
        # 注意：self.action_g_one_shape和self.action_k_one_shape同样应该相同
        # 疑问：在多智能体强化学习的编程框架中，要求每个智能体之间的状态和动作空间都一样吗？

        self.max_episode_steps = config.max_episode_steps  # 1000 #512*2000  # The maximum steps for one episode of the environment
        self.max_num_steps = config.max_num_steps  # 512*1000

        # 用于收敛性分析和评价指标
        self.dict_train_index = {'V': [], 'V_k1': [], 'V_k2': [], 'V_k3': [], 'V_g': [],
                                 'V_k1_0': [], 'V_k2_0': [], 'V_k3_0': [],
                                 'X_k1': [], 'X_k2': [], 'X_k3': [],
                                 'cost_k1': [], 'cost_k2': [], 'cost_k3': [],
                                 'revenue_k1': [], 'revenue_k2': [], 'revenue_k3': [], 'revenue_g': []}
        self.dict_eval_index = deepcopy(self.dict_train_index)

        self.eval_max_step = config.eval_max_step  # 512
        self.eval_max_episode = config.eval_max_episode  # 1
        self.eval_step_count = 0
        self.eval_episode_count = 0
        self.is_eval = False

        self.counts_total_reset = 0
        self.counts_episode = 0
        self.timestep = 0
        self.sample_idx = 0
        self.fixed_tm_idx = True # False # True

        self.reward_k1_list = []
        self.reward_k2_list = []
        self.reward_k3_list = []
        self.reward_g_list = []
        self.g_done_list = []

        self.observations = None
        self.infos = None

        ##**## 取出初始网络状态, 该部分代码由
        self.k1_net, self.k2_net, self.k3_net, self.g_net, self.DG = self.sample_dict[self.sample_idx]

        idx_k1_overlap_nodes, idx_k1_overlap_links, _ = self._get_overlap_node_link(k_net=self.k1_net, g_net=self.g_net)
        idx_k2_overlap_nodes, idx_k2_overlap_links, _ = self._get_overlap_node_link(k_net=self.k2_net, g_net=self.g_net)
        idx_k3_overlap_nodes, idx_k3_overlap_links, _ = self._get_overlap_node_link(k_net=self.k3_net, g_net=self.g_net)

        idx_k1_overlap_nodes = sorted(idx_k1_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k2_overlap_nodes = sorted(idx_k2_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k3_overlap_nodes = sorted(idx_k3_overlap_nodes)  # ！ 按照从小到大的顺序排列

        idx_k1_overlap_links = sorted(idx_k1_overlap_links)  # ！ 按照从小到大的顺序排列
        idx_k2_overlap_links = sorted(idx_k2_overlap_links)  # ！ 按照从小到大的顺序排列
        idx_k3_overlap_links = sorted(idx_k3_overlap_links)  # ！ 按照从小到大的顺序排列

        is_zero_padding = True
        obs_k1 = self._get_v_net_observations(self.k1_net, idx_k1_overlap_nodes, idx_k1_overlap_links, is_zero_padding)
        obs_k2 = self._get_v_net_observations(self.k2_net, idx_k2_overlap_nodes, idx_k2_overlap_links, is_zero_padding)
        obs_k3 = self._get_v_net_observations(self.k3_net, idx_k3_overlap_nodes, idx_k3_overlap_links, is_zero_padding)
        obs_g = self._get_g_net_observations(is_zero_padding)  # ！ 采用补零方案1

        self.observations = {self.agents[0]: obs_k1, self.agents[1]: obs_k2, self.agents[2]: obs_k3,
                             self.agents[-1]: obs_g}

        self.global_state = np.hstack((obs_k1, obs_k2, obs_k3, obs_g))

        # observations = {agent: self.observation_space[agent].sample() for agent in self.agents}
        # info = {}
        self.infos = {agent: {} for agent in self.agents}

        # 是否 eval
        # if self.is_eval == True:
        #     with open(self.path_env_save + '/' + self.save_attach_word + 'env_self.pkl', 'rb') as handle:
        #         self = pickle.load(handle)

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        return None

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        # return self.state_space.sample()
        return self.global_state

    def reset(self):
        logging.info("Reset the env once.")

        self.counts_total_reset += 1
        self.counts_episode = self.counts_episode + 1 if self.timestep > 0 else 0
        print('counts_total_reset = ' + str(self.counts_total_reset)
              + '; counts_episode = ' + str(self.counts_episode)
              + '; timestep = ' + str(self.timestep)
              + '; sample_idx = ' + str(self.sample_idx))

        # self.sample_idx = 0
        #
        # self.k1_net, self.k2_net, self.k3_net, self.g_net, self.DG = self.sample_dict[self.sample_idx]
        #
        # idx_k1_overlap_nodes, idx_k1_overlap_links, _ = self._get_overlap_node_link(k_net=self.k1_net, g_net=self.g_net)
        # idx_k2_overlap_nodes, idx_k2_overlap_links, _ = self._get_overlap_node_link(k_net=self.k2_net, g_net=self.g_net)
        # idx_k3_overlap_nodes, idx_k3_overlap_links, _ = self._get_overlap_node_link(k_net=self.k3_net, g_net=self.g_net)
        #
        # idx_k1_overlap_nodes = sorted(idx_k1_overlap_nodes)  # ！ 按照从小到大的顺序排列
        # idx_k2_overlap_nodes = sorted(idx_k2_overlap_nodes)  # ！ 按照从小到大的顺序排列
        # idx_k3_overlap_nodes = sorted(idx_k3_overlap_nodes)  # ！ 按照从小到大的顺序排列
        #
        # idx_k1_overlap_links = sorted(idx_k1_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k2_overlap_links = sorted(idx_k2_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k3_overlap_links = sorted(idx_k3_overlap_links)  # ！ 按照从小到大的顺序排列
        #
        # is_zero_padding = True
        # obs_k1 = self._get_v_net_observations(self.k1_net, idx_k1_overlap_nodes, idx_k1_overlap_links, is_zero_padding)
        # obs_k2 = self._get_v_net_observations(self.k2_net, idx_k2_overlap_nodes, idx_k2_overlap_links, is_zero_padding)
        # obs_k3 = self._get_v_net_observations(self.k3_net, idx_k3_overlap_nodes, idx_k3_overlap_links, is_zero_padding)
        # obs_g = self._get_g_net_observations(is_zero_padding)  #！ 采用补零方案1
        #
        # observations = {self.agents[0]: obs_k1, self.agents[1]: obs_k2, self.agents[2]: obs_k3,
        #                 self.agents[-1]: obs_g}
        #
        # self.global_state = np.hstack((obs_k1, obs_k2, obs_k3, obs_g))
        #
        # # observations = {agent: self.observation_space[agent].sample() for agent in self.agents}
        # # info = {}
        # infos = {agent: {} for agent in self.agents}
        # self.timestep = 0

        # return observations, infos

        return self.observations, self.infos

    def step(self, action_dict):
        logging.info("step = " + str(self.timestep))
        # start_time = time.time()

        if self.is_eval == True:
            return self.eval_env_step(action_dict)
        else:
            pass

        # 缩放到[0, 1]
        # actions = {key: np.clip(value, 0, 1) for key, value in action_dict.items()}
        actions = {key: np.clip((value + 1) / 2, 0, 1) for key, value in action_dict.items()}

        # 代表新生网络的智能体的动作输出是该新生网络每条链路和节点的资源占用率（占用多少物理网络资源），
        # 而代表已有网络的智能体的动作输出是具体需要为新生成网络而让出的物理资源（链路带宽和节点计算/存储等资源）。
        # 考虑到模型训练和收敛的问题，每个智能体的输出维度不应该过大
        # 方案1：每个智能体仅用两个神经元分别控制对应业务网络中所有与新生网络重叠链路的让出资源比和所有与新生网络重叠节点的让出资源比（一对所有）
        # 方案2：每个智能体用有限的神经元来控制相关的节点和链路的让出资源比，形成一个输出神经元控制固定个数的节点或链路（一对多），即粗粒度控制
        # 方案3：一对一控制，但需要考虑到维度爆炸的问题

        ##### k1 #####
        ## 代价计算
        # 输出 idx_k1_overlap_nodes和idx_k1_overlap_links
        idx_k1_overlap_nodes, idx_k1_overlap_links, sd_k_overlap_links = self._get_overlap_node_link(k_net=self.k1_net,
                                                                                                     g_net=self.g_net)
        idx_k1_overlap_nodes = sorted(idx_k1_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k1_overlap_links = sorted(idx_k1_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k1_overlap_nodes = [1, 2, 3, 9, 10]  # 按照idx排列，并于action_k1_node一一对应。
        action_k1 = actions['agent_k1']
        action_k1_node = action_k1[:self.action_k_node_control]  # 节点控制

        # 输入 action_k1_node  idx_k1_overlap_nodes
        # 输出 control_scale_n 和新的 action_k1_node # 一个神经元控制来多少节点元素
        action_k1_node, control_scale = self._get_scale_repeat_action(action_k1_node, idx_k1_overlap_nodes)
        old_demand_n_k1 = self._get_net_node_demand(v_net=self.k1_net)
        cost_n_k1, self.k1_net, assigned_comp_k1, assigned_cach_k1 = self._get_k_node_cost(k_net=self.k1_net,
                                                                                           idx_overlap_nodes=idx_k1_overlap_nodes,
                                                                                           action_k_node=action_k1_node)

        action_k1_link = action_k1[self.action_k_node_control:]  # 链路控制
        action_k1_link, control_scale = self._get_scale_repeat_action(action_k1_link, idx_k1_overlap_links)

        k1_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        cost_l_k1, self.k1_net, assigned_bandw_k1, demand_l_k1, old_demand_l_k1 = \
            self._get_k_link_cost(k_net=self.k1_net, idx_overlap_links=idx_k1_overlap_links,
                                  action_k_link=action_k1_link, tm_idx=k1_tm_idx)
        beta_n = 0.01 / 2
        beta_e = 1 / 5
        cost_k1 = beta_n * cost_n_k1 + beta_e * cost_l_k1

        ## 收益计算
        w_n = 0.01
        w_l = 10
        old_revenue_k1 = w_n * old_demand_n_k1 + w_l * old_demand_l_k1  # 在新生成业务网络之前，第k个已有网络所获得的来自任务服务的收益
        demand_n_k1 = self._get_net_node_demand(v_net=self.k1_net)
        revenue_k1 = w_n * demand_n_k1 + w_l * demand_l_k1  # 在新生成业务网络之后，第k个已有网络所获得的来自任务服务的收益

        ##### k2 #####
        ## 代价计算
        # 输出 idx_k2_overlap_nodes和idx_k2_overlap_links
        idx_k2_overlap_nodes, idx_k2_overlap_links, sd_k_overlap_links = self._get_overlap_node_link(k_net=self.k2_net,
                                                                                                     g_net=self.g_net)
        idx_k2_overlap_nodes = sorted(idx_k2_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k2_overlap_links = sorted(idx_k2_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k2_overlap_nodes = [1, 2, 3, 9, 10]  # 按照idx排列，并于action_k2_node一一对应。
        action_k2 = actions['agent_k2']
        action_k2_node = action_k2[:self.action_k_node_control]  # 节点控制
        action_k2_node, control_scale = self._get_scale_repeat_action(action_k2_node, idx_k2_overlap_nodes)
        old_demand_n_k2 = self._get_net_node_demand(v_net=self.k2_net)
        cost_n_k2, self.k2_net, assigned_comp_k2, assigned_cach_k2 = self._get_k_node_cost(k_net=self.k2_net,
                                                                                           idx_overlap_nodes=idx_k2_overlap_nodes,
                                                                                           action_k_node=action_k2_node)

        action_k2_link = action_k2[self.action_k_node_control:]  # 链路控制
        action_k2_link, control_scale = self._get_scale_repeat_action(action_k2_link, idx_k2_overlap_links)

        k2_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        cost_l_k2, self.k2_net, assigned_bandw_k2, demand_l_k2, old_demand_l_k2 = \
            self._get_k_link_cost(k_net=self.k2_net, idx_overlap_links=idx_k2_overlap_links,
                                  action_k_link=action_k2_link, tm_idx=k2_tm_idx)
        beta_n = 0.01 / 2
        beta_e = 1 / 5
        cost_k2 = beta_n * cost_n_k2 + beta_e * cost_l_k2

        ## 收益计算
        w_n = 0.01
        w_l = 10
        old_revenue_k2 = w_n * old_demand_n_k2 + w_l * old_demand_l_k2  # 在新生成业务网络之前，第k个已有网络所获得的来自任务服务的收益
        demand_n_k2 = self._get_net_node_demand(v_net=self.k2_net)
        revenue_k2 = w_n * demand_n_k2 + w_l * demand_l_k2  # 在新生成业务网络之后，第k个已有网络所获得的来自任务服务的收益

        ##### k3 #####
        ## 代价计算
        # 输出 idx_k3_overlap_nodes和idx_k3_overlap_links
        idx_k3_overlap_nodes, idx_k3_overlap_links, sd_k_overlap_links = self._get_overlap_node_link(k_net=self.k3_net,
                                                                                                     g_net=self.g_net)
        idx_k3_overlap_nodes = sorted(idx_k3_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k3_overlap_links = sorted(idx_k3_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k3_overlap_nodes = [1, 2, 3, 9, 10]  # 按照idx排列，并于action_k3_node一一对应。
        action_k3 = actions['agent_k3']
        action_k3_node = action_k3[:self.action_k_node_control]  # 节点控制
        action_k3_node, control_scale = self._get_scale_repeat_action(action_k3_node, idx_k3_overlap_nodes)
        old_demand_n_k3 = self._get_net_node_demand(v_net=self.k3_net)
        cost_n_k3, self.k3_net, assigned_comp_k3, assigned_cach_k3 = self._get_k_node_cost(k_net=self.k3_net,
                                                                                           idx_overlap_nodes=idx_k3_overlap_nodes,
                                                                                           action_k_node=action_k3_node)

        action_k3_link = action_k3[self.action_k_node_control:]  # 链路控制
        action_k3_link, control_scale = self._get_scale_repeat_action(action_k3_link, idx_k3_overlap_links)

        k3_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        cost_l_k3, self.k3_net, assigned_bandw_k3, demand_l_k3, old_demand_l_k3 = \
            self._get_k_link_cost(k_net=self.k3_net, idx_overlap_links=idx_k3_overlap_links,
                                  action_k_link=action_k3_link, tm_idx=k3_tm_idx)
        beta_n = 0.01 / 2
        beta_e = 1 / 5
        cost_k3 = beta_n * cost_n_k3 + beta_e * cost_l_k3

        ## 收益计算
        w_n = 0.01
        w_l = 10
        old_revenue_k3 = w_n * old_demand_n_k3 + w_l * old_demand_l_k3  # 在新生成业务网络之前，第k个已有网络所获得的来自任务服务的收益
        demand_n_k3 = self._get_net_node_demand(v_net=self.k3_net)
        revenue_k3 = w_n * demand_n_k3 + w_l * demand_l_k3  # 在新生成业务网络之后，第k个已有网络所获得的来自任务服务的收益

        ##### g #####
        assigned_resoure = {}
        assigned_resoure['k1_net'] = (assigned_comp_k1, assigned_cach_k1, assigned_bandw_k1)
        assigned_resoure['k2_net'] = (assigned_comp_k2, assigned_cach_k2, assigned_bandw_k2)
        assigned_resoure['k3_net'] = (assigned_comp_k3, assigned_cach_k3, assigned_bandw_k3)

        unassigned_resoure = self._get_g_idx_unassigned_resoure()  # 获取g_net各个节点和链路下，各个虚拟业务网络未让出资源之前的物理网络剩余未分配的资源

        g_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        action_g = actions['agent_g']  # [0.5, 0.5, 0.5]  # 分别代表计算、缓存、带宽资源的调整比例
        self.g_net, demand_n_g, demand_l_g, g_done = self._get_g_adjustment(g_net=self.g_net, action_g=action_g,
                                                                            assigned_resoure=assigned_resoure, tm_idx=g_tm_idx)
        if g_done == True:
            # g 的收益计算
            revenue_g = w_n * demand_n_g + w_l * demand_l_g
            #### 贡献度计算
            contribution_degree = self._get_contribution_degree(assigned_resoure=assigned_resoure,
                                                                unassigned_resoure=unassigned_resoure)
            #### X 计算, 根据贡献度
            X_k1 = revenue_g * contribution_degree[0]
            X_k2 = revenue_g * contribution_degree[1]
            X_k3 = revenue_g * contribution_degree[2]
        else:
            revenue_g = 0
            contribution_degree = np.zeros(self.K_num_existing_v_net + 1)
            X_k1 = 0
            X_k2 = 0
            X_k3 = 0

        #### 效益计算
        V_k1_0 = old_revenue_k1
        V_k1 = revenue_k1 - cost_k1 + X_k1

        V_k2_0 = old_revenue_k2
        V_k2 = revenue_k2 - cost_k2 + X_k2

        V_k3_0 = old_revenue_k3
        V_k3 = revenue_k3 - cost_k3 + X_k3

        V_g_0 = 0
        V_g = revenue_g - X_k1 - X_k2 - X_k3

        #### rewards
        reward_k1 = contribution_degree[0] * (V_k1 - V_k1_0) / (V_k1_0 + OBJ_EPSILON) * 10 * 4 if g_done == True else 0
        reward_k2 = contribution_degree[1] * (V_k2 - V_k2_0) / (V_k2_0 + OBJ_EPSILON) * 10 * 4 if g_done == True else 0
        reward_k3 = contribution_degree[2] * (V_k3 - V_k3_0) / (V_k3_0 + OBJ_EPSILON) * 10 * 4 if g_done == True else 0
        reward_g = contribution_degree[3] * (V_g - V_g_0) / 20 * 1.5 if g_done == True else 0
        # clip
        reward_k1 = np.clip(reward_k1, -5, 5)
        reward_k2 = np.clip(reward_k2, -5, 5)
        reward_k3 = np.clip(reward_k3, -5, 5)
        reward_g = np.clip(reward_g, -5, 5)

        rewards = {self.agents[0]: reward_k1, self.agents[1]: reward_k2, self.agents[2]: reward_k3,
                   self.agents[-1]: reward_g}

        ##!!
        # rewards = {self.agents[0]: random.random(), self.agents[1]: random.random(), self.agents[2]: random.random(),
        #            self.agents[-1]: random.random()}

        # 保存 rewards
        self.reward_k1_list.append(reward_k1)
        self.reward_k2_list.append(reward_k2)
        self.reward_k3_list.append(reward_k3)
        self.reward_g_list.append(reward_g)
        self.g_done_list.append(g_done)

        if self.timestep + 1 == self.max_num_steps:

            # attach_word = str(self.timestep + 1) + '_' + self.topology_file + '_'

            with open(self.path_reward_save + '/' + self.save_attach_word + 'rewards_marl_env.pkl',
                      'wb') as handle:
                pickle.dump((self.reward_k1_list, self.reward_k2_list, self.reward_k3_list,
                             self.reward_g_list, self.g_done_list), handle)

            self.is_eval = True  ## 之后开始eval_env_step

        if (self.timestep + 1) % 10000 == 0:
            with open(self.path_reward_save + '/' + self.topology_file + '_rewards_temper_marl_env.pkl',
                      'wb') as handle:
                pickle.dump((self.reward_k1_list, self.reward_k2_list, self.reward_k3_list,
                             self.reward_g_list, self.g_done_list), handle)

        #*# 用于收敛性分析和评价指标
        self.dict_train_index['V'].append(V_k1 + V_k2 + V_k3 + V_g)  # <1> 总效益 V
        self.dict_train_index['V_k1'].append(V_k1)  # <1> K1 效益 V_k1
        self.dict_train_index['V_k2'].append(V_k2)  # <1> K2 效益 V_k2
        self.dict_train_index['V_k3'].append(V_k3)  # <1> K3 效益 V_k3
        self.dict_train_index['V_g'].append(V_g)  # <1> g 效益 V_g
        self.dict_train_index['V_k1_0'].append(V_k1_0)
        self.dict_train_index['V_k2_0'].append(V_k2_0)
        self.dict_train_index['V_k3_0'].append(V_k3_0)
        self.dict_train_index['X_k1'].append(X_k1)
        self.dict_train_index['X_k2'].append(X_k2)
        self.dict_train_index['X_k3'].append(X_k3)
        self.dict_train_index['cost_k1'].append(cost_k1)
        self.dict_train_index['cost_k2'].append(cost_k2)
        self.dict_train_index['cost_k3'].append(cost_k3)
        self.dict_train_index['revenue_k1'].append(revenue_k1)
        self.dict_train_index['revenue_k2'].append(revenue_k2)
        self.dict_train_index['revenue_k3'].append(revenue_k3)
        self.dict_train_index['revenue_g'].append(revenue_g)

        if self.timestep + 1 == self.max_num_steps:
            with open(self.path_reward_save + '/' + self.save_attach_word + 'dict_train_index_marl_env.pkl',
                      'wb') as handle:
                pickle.dump(self.dict_train_index, handle)
        #*#

        #### observations
        # 更新 sample_idx
        if self.sample_idx + 1 < self.sample_cnt:
            self.sample_idx = self.sample_idx + 1
        else:
            self.sample_idx = 0
            with open(self.sample_dict_save_file, 'rb') as pkl:
                self.sample_dict = pickle.load(pkl)

        self.k1_net, self.k2_net, self.k3_net, self.g_net, self.DG = self.sample_dict[self.sample_idx]

        # # 检查
        # allocated_bandw_list = []
        # bandwidth_list = []
        # for idx_link in range(self.num_links):
        #     s, d = self.link_idx_to_sd[idx_link]
        #     bandwidth_list.append(self.DG[s][d]['bandwidth'])  # Gbps
        #     allocated_bandw_list.append(self.DG[s][d]['allocated_bandw'])
        # if max(allocated_bandw_list) > 33:
        #     print("Error")

        idx_k1_overlap_nodes, idx_k1_overlap_links, _ = self._get_overlap_node_link(k_net=self.k1_net, g_net=self.g_net)
        idx_k2_overlap_nodes, idx_k2_overlap_links, _ = self._get_overlap_node_link(k_net=self.k2_net, g_net=self.g_net)
        idx_k3_overlap_nodes, idx_k3_overlap_links, _ = self._get_overlap_node_link(k_net=self.k3_net, g_net=self.g_net)

        is_zero_padding = True
        obs_k1 = self._get_v_net_observations(self.k1_net, idx_k1_overlap_nodes, idx_k1_overlap_links, is_zero_padding)
        obs_k2 = self._get_v_net_observations(self.k2_net, idx_k2_overlap_nodes, idx_k2_overlap_links, is_zero_padding)
        obs_k3 = self._get_v_net_observations(self.k3_net, idx_k3_overlap_nodes, idx_k3_overlap_links, is_zero_padding)
        obs_g = self._get_g_net_observations(is_zero_padding)  # ！

        observations = {self.agents[0]: obs_k1, self.agents[1]: obs_k2, self.agents[2]: obs_k3,
                        self.agents[-1]: obs_g}

        # self.global_state = np.hstack((obs_k1, obs_k2, obs_k3, obs_g))

        # !!####
        # observations = {agent: self.observation_space[agent].sample() for agent in self.agents}
        # rewards = {agent: np.random.random() for agent in self.agents}

        self.timestep += 1

        terminated = {agent: False for agent in self.agents}
        truncated = False if self.timestep % self.max_episode_steps != 0 else True
        # truncated = False if self.timestep % self.max_num_steps != 0 else True
        infos = {agent: {} for agent in self.agents}

        # end_time = time.time()
        # running_time = end_time - start_time
        # print("测试代码'step'的运行时间：", running_time * 1000, "ms")
        self.observations = observations
        self.infos = infos

        # # 保存env self
        # if self.timestep == self.max_num_steps:
        #     with open(self.path_env_save + '/' + self.save_attach_word + 'env_self.pkl', 'wb') as handle:
        #         pickle.dump((self), handle)

        return observations, rewards, terminated, truncated, infos

    def eval_env_step(self, action_dict):
        logging.info("timestep = " + str(self.timestep) + "; eval_step_count = " + str(self.eval_step_count))
        # start_time = time.time()

        # 缩放到[0, 1]
        # actions = {key: np.clip(value, 0, 1) for key, value in action_dict.items()}
        actions = {key: np.clip((value + 1) / 2, 0, 1) for key, value in action_dict.items()}

        # 代表新生网络的智能体的动作输出是该新生网络每条链路和节点的资源占用率（占用多少物理网络资源），
        # 而代表已有网络的智能体的动作输出是具体需要为新生成网络而让出的物理资源（链路带宽和节点计算/存储等资源）。
        # 考虑到模型训练和收敛的问题，每个智能体的输出维度不应该过大
        # 方案1：每个智能体仅用两个神经元分别控制对应业务网络中所有与新生网络重叠链路的让出资源比和所有与新生网络重叠节点的让出资源比（一对所有）
        # 方案2：每个智能体用有限的神经元来控制相关的节点和链路的让出资源比，形成一个输出神经元控制固定个数的节点或链路（一对多），即粗粒度控制
        # 方案3：一对一控制，但需要考虑到维度爆炸的问题

        ##### k1 #####
        ## 代价计算
        # 输出 idx_k1_overlap_nodes和idx_k1_overlap_links
        idx_k1_overlap_nodes, idx_k1_overlap_links, sd_k_overlap_links = self._get_overlap_node_link(k_net=self.k1_net,
                                                                                                     g_net=self.g_net)
        idx_k1_overlap_nodes = sorted(idx_k1_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k1_overlap_links = sorted(idx_k1_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k1_overlap_nodes = [1, 2, 3, 9, 10]  # 按照idx排列，并于action_k1_node一一对应。
        action_k1 = actions['agent_k1']
        action_k1_node = action_k1[:self.action_k_node_control]  # 节点控制

        # 输入 action_k1_node  idx_k1_overlap_nodes
        # 输出 control_scale_n 和新的 action_k1_node # 一个神经元控制来多少节点元素
        action_k1_node, control_scale = self._get_scale_repeat_action(action_k1_node, idx_k1_overlap_nodes)
        old_demand_n_k1 = self._get_net_node_demand(v_net=self.k1_net)
        cost_n_k1, self.k1_net, assigned_comp_k1, assigned_cach_k1 = self._get_k_node_cost(k_net=self.k1_net,
                                                                                           idx_overlap_nodes=idx_k1_overlap_nodes,
                                                                                           action_k_node=action_k1_node)

        action_k1_link = action_k1[self.action_k_node_control:]  # 链路控制
        action_k1_link, control_scale = self._get_scale_repeat_action(action_k1_link, idx_k1_overlap_links)

        k1_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        cost_l_k1, self.k1_net, assigned_bandw_k1, demand_l_k1, old_demand_l_k1 = \
            self._get_k_link_cost(k_net=self.k1_net, idx_overlap_links=idx_k1_overlap_links,
                                  action_k_link=action_k1_link, tm_idx=k1_tm_idx)
        beta_n = 0.01 / 2
        beta_e = 1 / 5
        cost_k1 = beta_n * cost_n_k1 + beta_e * cost_l_k1

        ## 收益计算
        w_n = 0.01
        w_l = 10
        old_revenue_k1 = w_n * old_demand_n_k1 + w_l * old_demand_l_k1  # 在新生成业务网络之前，第k个已有网络所获得的来自任务服务的收益
        demand_n_k1 = self._get_net_node_demand(v_net=self.k1_net)
        revenue_k1 = w_n * demand_n_k1 + w_l * demand_l_k1  # 在新生成业务网络之后，第k个已有网络所获得的来自任务服务的收益

        ##### k2 #####
        ## 代价计算
        # 输出 idx_k2_overlap_nodes和idx_k2_overlap_links
        idx_k2_overlap_nodes, idx_k2_overlap_links, sd_k_overlap_links = self._get_overlap_node_link(k_net=self.k2_net,
                                                                                                     g_net=self.g_net)
        idx_k2_overlap_nodes = sorted(idx_k2_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k2_overlap_links = sorted(idx_k2_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k2_overlap_nodes = [1, 2, 3, 9, 10]  # 按照idx排列，并于action_k2_node一一对应。
        action_k2 = actions['agent_k2']
        action_k2_node = action_k2[:self.action_k_node_control]  # 节点控制
        action_k2_node, control_scale = self._get_scale_repeat_action(action_k2_node, idx_k2_overlap_nodes)
        old_demand_n_k2 = self._get_net_node_demand(v_net=self.k2_net)
        cost_n_k2, self.k2_net, assigned_comp_k2, assigned_cach_k2 = self._get_k_node_cost(k_net=self.k2_net,
                                                                                           idx_overlap_nodes=idx_k2_overlap_nodes,
                                                                                           action_k_node=action_k2_node)

        action_k2_link = action_k2[self.action_k_node_control:]  # 链路控制
        action_k2_link, control_scale = self._get_scale_repeat_action(action_k2_link, idx_k2_overlap_links)

        k2_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        cost_l_k2, self.k2_net, assigned_bandw_k2, demand_l_k2, old_demand_l_k2 = \
            self._get_k_link_cost(k_net=self.k2_net, idx_overlap_links=idx_k2_overlap_links,
                                  action_k_link=action_k2_link, tm_idx=k2_tm_idx)
        beta_n = 0.01 / 2
        beta_e = 1 / 5
        cost_k2 = beta_n * cost_n_k2 + beta_e * cost_l_k2

        ## 收益计算
        w_n = 0.01
        w_l = 10
        old_revenue_k2 = w_n * old_demand_n_k2 + w_l * old_demand_l_k2  # 在新生成业务网络之前，第k个已有网络所获得的来自任务服务的收益
        demand_n_k2 = self._get_net_node_demand(v_net=self.k2_net)
        revenue_k2 = w_n * demand_n_k2 + w_l * demand_l_k2  # 在新生成业务网络之后，第k个已有网络所获得的来自任务服务的收益

        ##### k3 #####
        ## 代价计算
        # 输出 idx_k3_overlap_nodes和idx_k3_overlap_links
        idx_k3_overlap_nodes, idx_k3_overlap_links, sd_k_overlap_links = self._get_overlap_node_link(k_net=self.k3_net,
                                                                                                     g_net=self.g_net)
        idx_k3_overlap_nodes = sorted(idx_k3_overlap_nodes)  # ！ 按照从小到大的顺序排列
        idx_k3_overlap_links = sorted(idx_k3_overlap_links)  # ！ 按照从小到大的顺序排列
        # idx_k3_overlap_nodes = [1, 2, 3, 9, 10]  # 按照idx排列，并于action_k3_node一一对应。
        action_k3 = actions['agent_k3']
        action_k3_node = action_k3[:self.action_k_node_control]  # 节点控制
        action_k3_node, control_scale = self._get_scale_repeat_action(action_k3_node, idx_k3_overlap_nodes)
        old_demand_n_k3 = self._get_net_node_demand(v_net=self.k3_net)
        cost_n_k3, self.k3_net, assigned_comp_k3, assigned_cach_k3 = self._get_k_node_cost(k_net=self.k3_net,
                                                                                           idx_overlap_nodes=idx_k3_overlap_nodes,
                                                                                           action_k_node=action_k3_node)

        action_k3_link = action_k3[self.action_k_node_control:]  # 链路控制
        action_k3_link, control_scale = self._get_scale_repeat_action(action_k3_link, idx_k3_overlap_links)

        k3_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        cost_l_k3, self.k3_net, assigned_bandw_k3, demand_l_k3, old_demand_l_k3 = \
            self._get_k_link_cost(k_net=self.k3_net, idx_overlap_links=idx_k3_overlap_links,
                                  action_k_link=action_k3_link, tm_idx=k3_tm_idx)
        beta_n = 0.01 / 2
        beta_e = 1 / 5
        cost_k3 = beta_n * cost_n_k3 + beta_e * cost_l_k3

        ## 收益计算
        w_n = 0.01
        w_l = 10
        old_revenue_k3 = w_n * old_demand_n_k3 + w_l * old_demand_l_k3  # 在新生成业务网络之前，第k个已有网络所获得的来自任务服务的收益
        demand_n_k3 = self._get_net_node_demand(v_net=self.k3_net)
        revenue_k3 = w_n * demand_n_k3 + w_l * demand_l_k3  # 在新生成业务网络之后，第k个已有网络所获得的来自任务服务的收益

        ##### g #####
        assigned_resoure = {}
        assigned_resoure['k1_net'] = (assigned_comp_k1, assigned_cach_k1, assigned_bandw_k1)
        assigned_resoure['k2_net'] = (assigned_comp_k2, assigned_cach_k2, assigned_bandw_k2)
        assigned_resoure['k3_net'] = (assigned_comp_k3, assigned_cach_k3, assigned_bandw_k3)

        unassigned_resoure = self._get_g_idx_unassigned_resoure()  # 获取g_net各个节点和链路下，各个虚拟业务网络未让出资源之前的物理网络剩余未分配的资源

        g_tm_idx = 0 if self.fixed_tm_idx == True else self.sample_idx
        action_g = actions['agent_g']  # [0.5, 0.5, 0.5]  # 分别代表计算、缓存、带宽资源的调整比例
        self.g_net, demand_n_g, demand_l_g, g_done = self._get_g_adjustment(g_net=self.g_net, action_g=action_g,
                                                                            assigned_resoure=assigned_resoure, tm_idx=g_tm_idx)
        if g_done == True:
            # g 的收益计算
            revenue_g = w_n * demand_n_g + w_l * demand_l_g
            #### 贡献度计算
            contribution_degree = self._get_contribution_degree(assigned_resoure=assigned_resoure,
                                                                unassigned_resoure=unassigned_resoure)
            #### X 计算, 根据贡献度
            X_k1 = revenue_g * contribution_degree[0]
            X_k2 = revenue_g * contribution_degree[1]
            X_k3 = revenue_g * contribution_degree[2]
        else:
            revenue_g = 0
            contribution_degree = np.zeros(self.K_num_existing_v_net + 1)
            X_k1 = 0
            X_k2 = 0
            X_k3 = 0

        #### 效益计算
        V_k1_0 = old_revenue_k1
        V_k1 = revenue_k1 - cost_k1 + X_k1

        V_k2_0 = old_revenue_k2
        V_k2 = revenue_k2 - cost_k2 + X_k2

        V_k3_0 = old_revenue_k3
        V_k3 = revenue_k3 - cost_k3 + X_k3

        V_g_0 = 0
        V_g = revenue_g - X_k1 - X_k2 - X_k3

        #### rewards
        reward_k1 = contribution_degree[0] * (V_k1 - V_k1_0) / (V_k1_0 + OBJ_EPSILON) * 10 * 4 if g_done == True else 0
        reward_k2 = contribution_degree[1] * (V_k2 - V_k2_0) / (V_k2_0 + OBJ_EPSILON) * 10 * 4 if g_done == True else 0
        reward_k3 = contribution_degree[2] * (V_k3 - V_k3_0) / (V_k3_0 + OBJ_EPSILON) * 10 * 4 if g_done == True else 0
        reward_g = contribution_degree[3] * (V_g - V_g_0) / 20 * 1.5 if g_done == True else 0
        # clip
        reward_k1 = np.clip(reward_k1, -5, 5)
        reward_k2 = np.clip(reward_k2, -5, 5)
        reward_k3 = np.clip(reward_k3, -5, 5)
        reward_g = np.clip(reward_g, -5, 5)

        rewards = {self.agents[0]: reward_k1, self.agents[1]: reward_k2, self.agents[2]: reward_k3,
                   self.agents[-1]: reward_g}

        ##!!
        # rewards = {self.agents[0]: random.random(), self.agents[1]: random.random(), self.agents[2]: random.random(),
        #            self.agents[-1]: random.random()}

        # 保存 rewards
        self.reward_k1_list.append(reward_k1)
        self.reward_k2_list.append(reward_k2)
        self.reward_k3_list.append(reward_k3)
        self.reward_g_list.append(reward_g)
        self.g_done_list.append(g_done)

        # if self.timestep + 1 == self.max_num_steps:
        #     with open(self.path_reward_save + '/' + str(
        #             self.timestep + 1) + '_' + self.topology_file + '_rewards_marl_env.pkl',
        #               'wb') as handle:
        #         pickle.dump((self.reward_k1_list, self.reward_k2_list, self.reward_k3_list,
        #                      self.reward_g_list, self.g_done_list), handle)
        #
        # if (self.timestep + 1) % 10000 == 0:
        #     with open(self.path_reward_save + '/' + self.topology_file + '_rewards_temper_marl_env.pkl',
        #               'wb') as handle:
        #         pickle.dump((self.reward_k1_list, self.reward_k2_list, self.reward_k3_list,
        #                      self.reward_g_list, self.g_done_list), handle)

        # *# 用于评价指标分析
        self.dict_eval_index['V'].append(V_k1 + V_k2 + V_k3 + V_g)  # <1> 总效益 V
        self.dict_eval_index['V_k1'].append(V_k1)  # <1> K1 效益 V_k1
        self.dict_eval_index['V_k2'].append(V_k2)  # <1> K2 效益 V_k2
        self.dict_eval_index['V_k3'].append(V_k3)  # <1> K3 效益 V_k3
        self.dict_eval_index['V_g'].append(V_g)  # <1> g 效益 V_g
        self.dict_eval_index['V_k1_0'].append(V_k1_0)
        self.dict_eval_index['V_k2_0'].append(V_k2_0)
        self.dict_eval_index['V_k3_0'].append(V_k3_0)
        self.dict_eval_index['X_k1'].append(X_k1)
        self.dict_eval_index['X_k2'].append(X_k2)
        self.dict_eval_index['X_k3'].append(X_k3)
        self.dict_eval_index['cost_k1'].append(cost_k1)
        self.dict_eval_index['cost_k2'].append(cost_k2)
        self.dict_eval_index['cost_k3'].append(cost_k3)
        self.dict_eval_index['revenue_k1'].append(revenue_k1)
        self.dict_eval_index['revenue_k2'].append(revenue_k2)
        self.dict_eval_index['revenue_k3'].append(revenue_k3)
        self.dict_eval_index['revenue_g'].append(revenue_g)
        #*#

        #### observations
        # 更新 sample_idx
        if self.sample_idx + 1 < self.sample_cnt:
            self.sample_idx = self.sample_idx + 1
        else:
            self.sample_idx = 0
            with open(self.sample_dict_save_file, 'rb') as pkl:
                self.sample_dict = pickle.load(pkl)

        self.k1_net, self.k2_net, self.k3_net, self.g_net, self.DG = self.sample_dict[self.sample_idx]

        # # 检查
        # allocated_bandw_list = []
        # bandwidth_list = []
        # for idx_link in range(self.num_links):
        #     s, d = self.link_idx_to_sd[idx_link]
        #     bandwidth_list.append(self.DG[s][d]['bandwidth'])  # Gbps
        #     allocated_bandw_list.append(self.DG[s][d]['allocated_bandw'])
        # if max(allocated_bandw_list) > 33:
        #     print("Error")

        idx_k1_overlap_nodes, idx_k1_overlap_links, _ = self._get_overlap_node_link(k_net=self.k1_net, g_net=self.g_net)
        idx_k2_overlap_nodes, idx_k2_overlap_links, _ = self._get_overlap_node_link(k_net=self.k2_net, g_net=self.g_net)
        idx_k3_overlap_nodes, idx_k3_overlap_links, _ = self._get_overlap_node_link(k_net=self.k3_net, g_net=self.g_net)

        is_zero_padding = True
        obs_k1 = self._get_v_net_observations(self.k1_net, idx_k1_overlap_nodes, idx_k1_overlap_links, is_zero_padding)
        obs_k2 = self._get_v_net_observations(self.k2_net, idx_k2_overlap_nodes, idx_k2_overlap_links, is_zero_padding)
        obs_k3 = self._get_v_net_observations(self.k3_net, idx_k3_overlap_nodes, idx_k3_overlap_links, is_zero_padding)
        obs_g = self._get_g_net_observations(is_zero_padding)  # ！

        observations = {self.agents[0]: obs_k1, self.agents[1]: obs_k2, self.agents[2]: obs_k3,
                        self.agents[-1]: obs_g}

        # self.global_state = np.hstack((obs_k1, obs_k2, obs_k3, obs_g))

        # !!####
        # observations = {agent: self.observation_space[agent].sample() for agent in self.agents}
        # rewards = {agent: np.random.random() for agent in self.agents}

        self.timestep += 1
        self.eval_step_count += 1

        terminated = {agent: False for agent in self.agents}
        # truncated = False if self.timestep % self.max_episode_steps != 0 else True

        # ## 方式1
        # truncated = False if self.eval_step_count < self.eval_max_step else True
        # if self.eval_step_count % self.max_episode_steps == 0:
        #     self.eval_episode_count += 1
        # if truncated == False:
        #     if self.eval_episode_count >= self.eval_max_episode:
        #         truncated = True
        #
        # else:
        #     with open(self.path_reward_save + '/' + 'eval_marl_env.pkl',
        #               'wb') as handle:
        #         pickle.dump((self.reward_k1_list, self.reward_k2_list, self.reward_k3_list,
        #                      self.reward_g_list, self.g_done_list), handle)

        ## 方式2
        truncated = False # if self.eval_step_count < self.eval_max_step else True
        if self.eval_step_count % self.eval_max_step == 0:
            self.eval_episode_count += 1
            with open(self.path_reward_save + '/' + self.save_attach_word + 'eval_marl_env.pkl', 'wb') as handle:
                pickle.dump((self.reward_k1_list, self.reward_k2_list, self.reward_k3_list,
                             self.reward_g_list, self.g_done_list, self.dict_eval_index), handle)


        # truncated = False if self.timestep % self.max_num_steps != 0 else True
        infos = {agent: {} for agent in self.agents}

        # end_time = time.time()
        # running_time = end_time - start_time
        # print("测试代码'step'的运行时间：", running_time * 1000, "ms")
        self.observations = observations
        self.infos = infos

        return observations, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        return

    def _init_p_net_topology_traffic(self, config):

        if config.topology_file == 'Abilene':  # *#
            self.topology = AbileneTopology(config, self.data_dir)
            self.traffic = AbileneTraffic(config, self.topology.num_nodes, self.data_dir)
        elif config.topology_file == 'GEANT':  # *#
            self.topology = GEANTTopology(config, self.data_dir)
            self.traffic = GEANTTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Xspedius':  # *#
            self.topology = utlis.network_env.XspediusTopology(config, self.data_dir)
            self.traffic = utlis.network_env.XspediusTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'SwitchL3':  # *#
            self.topology = utlis.network_env.SwitchL3Topology(config, self.data_dir)
            self.traffic = utlis.network_env.SwitchL3Traffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Uunet':  # *#
            self.topology = utlis.network_env.UunetTopology(config, self.data_dir)
            self.traffic = utlis.network_env.UunetTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Dfn':
            self.topology = utlis.network_env.DfnTopology(config, self.data_dir)
            self.traffic = utlis.network_env.DfnTraffic(config, self.topology, self.data_dir)
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
        ##
        self.neighbors_idx = {i: list(self.DG.neighbors(i)) for i in range(self.num_nodes)}  # 邻居节点
        self.neighbors_idx_2_order = {}  # 二阶邻居节点
        for key, value in self.neighbors_idx.items():
            neighbors_2o = []
            for i_2o in value:
                neighbors_2o = neighbors_2o + self.neighbors_idx[i_2o]
            nodes_exc = [key] + value
            neighbors_2o = [x for x in neighbors_2o if x not in nodes_exc]  # 剔除源节点和一阶邻居节点
            # 剔除重复出现的数字，并从小到大排序
            unique_elements = []
            seen = set()
            for x in neighbors_2o:
                if x not in seen:
                    unique_elements.append(x)
                    seen.add(x)
            self.neighbors_idx_2_order[key] = sorted(unique_elements)

        self.node_id_to_idx = self.topology.node_id_to_idx
        self.node_id_list = self.topology.node_id_list
        self.link_id_list = self.topology.link_id_list  ## !注意links乱序，不建议使用
        self.link_idx_list = self.topology.link_idx_list  ## !注意links乱序，不建议使用

        self._get_ecmp_next_hops()  # self.ecmp_next_hops

        self.flow_pairs = [p for p in range(self.num_pairs)]

        self.load_multiplier = {}

    def _set_v_network_attr(self, v_net, comp_capacity=300, cach_capacity=150, bandwidth=10):
        # 节点设置
        for i in range(v_net.num_nodes):
            # comp_capacity = 300
            # cach_capacity = 150
            v_net.DG.nodes[i]['router_performance'] = np.clip(np.random.normal(loc=1, scale=0.1), 0.9, 1.1)
            v_net.DG.nodes[i]['original_comp_capacity'] = comp_capacity
            v_net.DG.nodes[i]['original_cach_capacity'] = cach_capacity
            v_net.DG.nodes[i]['comp_capacity'] = comp_capacity  # 节点的计算资源总容量
            v_net.DG.nodes[i]['cach_capacity'] = cach_capacity  # 节点的存储资源总容量
            # comp_effic = np.clip(np.random.normal(loc=1, scale=0.2), 0.8, 1.2)
            comp_effic = np.random.uniform(0.8, 1.2)
            v_net.DG.nodes[i]['comp_effic'] = comp_effic  # 节点的计算效率

            num_node_tasks = np.random.poisson(lam=40)  # 任务的数量服从单位时间内的平均次数（率）为lam的泊松分布
            tasks_info_list, num_cach_tasks = self._task_random_uniform_generation(task_num=num_node_tasks,
                                                                                   comp_low=6, comp_high=10,
                                                                                   cach_low=3, cach_high=5)
            num_comp_tasks = num_node_tasks - num_cach_tasks
            # self._task_random_normal_generation
            # 每个节点运行（或者存储）一定数量的计算/缓存任务，并且为确保QoS和系统运行的稳定与高效，考虑每个节点的资源利用率设定一个上限u_lim（i.e.,0.8,0.9,1），
            # 超过该上限值就需要迁移一部分任务到其他的节点中运行，因此，任务

            # 计算出节点中已使用的计算资源和缓存资源
            used_comp = 0
            used_cach = 0
            for i_task in range(len(tasks_info_list)):
                used_comp = used_comp + tasks_info_list[i_task]['dem_comp']
                used_cach = used_cach + tasks_info_list[i_task]['dem_cach']

            # 判断节点中计算/缓存资源是否超出总量，若超出则弹出tasks_info_list末尾的任务（这里认为list右端为末尾，且新任务的task_info从末尾压入该list）
            # 计算资源满，则弹出仅计算任务；存储资源满，则依次弹出计算/缓存任务
            while used_comp > comp_capacity or used_cach > cach_capacity:
                if len(tasks_info_list) == 0:
                    break
                if used_comp > comp_capacity:  # 计算资源满，则仅弹出计算任务
                    for i_rt in range(len(tasks_info_list)):  # 从tasks_info_list找出排的最靠后的计算任务，并弹出它
                        if tasks_info_list[-(i_rt + 1)]['type'] == 'comp':
                            # 更新出节点中已使用的计算资源和缓存资源
                            mig_task = tasks_info_list[-(i_rt + 1)]  # 被（迁移）弹出的task
                            used_comp = used_comp - mig_task['dem_comp']
                            used_cach = used_cach - mig_task['dem_cach']
                            tasks_info_list = tasks_info_list[:-(i_rt + 1)] + tasks_info_list[
                                                                              -(i_rt + 1) + 1:] if i_rt != 0 \
                                else tasks_info_list[:-1]  # 弹出一个task
                            num_node_tasks -= 1
                            num_comp_tasks -= 1
                            break
                elif used_cach > cach_capacity:  # 缓存资源满，则弹出list最末尾的一个计算或者缓存任务
                    # 更新出节点中已使用的计算资源和缓存资源
                    mig_task = tasks_info_list[-1]  # 被（迁移）弹出的task
                    used_comp = used_comp - mig_task['dem_comp']
                    used_cach = used_cach - mig_task['dem_cach']
                    tasks_info_list = tasks_info_list[:-1]
                    num_node_tasks -= 1
                    if mig_task['type'] == 'comp':
                        num_comp_tasks -= 1
                    elif mig_task['type'] == 'cach':
                        num_cach_tasks -= 1
                else:
                    pass
                if used_comp <= comp_capacity and used_cach <= cach_capacity:
                    break

            # num_node_tasks = len(tasks_info_list)
            assert len(
                tasks_info_list) == num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            v_net.DG.nodes[i]['num_node_tasks'] = num_node_tasks  # 节点的任务数量
            v_net.DG.nodes[i]['num_comp_tasks'] = num_comp_tasks  # 节点的计算任务数量
            v_net.DG.nodes[i]['num_cach_tasks'] = num_cach_tasks  # 节点的缓存任务数量
            v_net.DG.nodes[i]['tasks_info_list'] = tasks_info_list  # 更新节点的tasks_info_list
            v_net.DG.nodes[i]['used_comp'] = used_comp  # 更新节点中已使用的计算资源
            v_net.DG.nodes[i]['used_cach'] = used_cach  # 更新节点中已使用的存储资源

            # 更新基础物理网络
            id_node = v_net.node_id_list[i]
            idx_p_net = self.node_id_to_idx[id_node]
            self.DG.nodes[idx_p_net]['allocated_comp'] += v_net.DG.nodes[i]['comp_capacity']  # 更新物理网络节点中已分配的计算资源
            self.DG.nodes[idx_p_net]['allocated_cach'] += v_net.DG.nodes[i]['cach_capacity']  # 更新物理网络节点中已分配的存储资源

        # 链路设置
        for i in range(v_net.num_links):
            s, d = v_net.link_idx_to_sd[i]
            v_net.link_capacities = bandwidth  # 10 # Gbps #重新修改初始的链路容量参数
            v_net.DG[s][d]['bandwidth'] = bandwidth  # 10 # Gbps
            v_net.DG[s][d]['original_bandwidth'] = bandwidth  # 10 # Gbps
            v_net.DG[s][d]['used_bandw'] = 0  # Gbps
            # 假定链路可以专门为一个任务迁移的数据流传输分配其总带宽的 1/32

            # 更新基础物理网络
            id_s = v_net.node_id_list[s]
            id_d = v_net.node_id_list[d]
            idx_s_p_net = self.node_id_to_idx[id_s]
            idx_d_p_net = self.node_id_to_idx[id_d]
            self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] += v_net.DG[s][d]['bandwidth']  # 物理网络链路中已分配的带宽资源
            if self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] > self.DG[idx_s_p_net][idx_d_p_net]['bandwidth']:
                print("Error")
            assert self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] <= self.DG[idx_s_p_net][idx_d_p_net][
                'bandwidth'], \
                "Error from the allocated_bandw."

        return v_net

    def _set_g_network_attr(self, g_net, comp_capacity=300, cach_capacity=150, bandwidth=10):
        # 实验设定 在新生成的网络中，comp_capacity, cach_capacity, bandwidth这三种资源在每个节点和链路中都是相同的
        # 节点设置
        for i in range(g_net.num_nodes):
            # comp_capacity = 300
            # cach_capacity = 150
            g_net.DG.nodes[i]['router_performance'] = np.clip(np.random.normal(loc=1, scale=0.1), 0.9, 1.1)
            g_net.DG.nodes[i]['original_comp_capacity'] = comp_capacity
            g_net.DG.nodes[i]['original_cach_capacity'] = cach_capacity
            g_net.DG.nodes[i]['comp_capacity'] = comp_capacity  # 节点的计算资源总容量
            g_net.DG.nodes[i]['cach_capacity'] = cach_capacity  # 节点的存储资源总容量
            # comp_effic = np.clip(np.random.normal(loc=1, scale=0.2), 0.8, 1.2)
            comp_effic = np.random.uniform(0.8, 1.2)
            g_net.DG.nodes[i]['comp_effic'] = comp_effic  # 节点的计算效率

            num_node_tasks = np.random.poisson(lam=40)  # 任务的数量服从单位时间内的平均次数（率）为lam的泊松分布
            tasks_info_list, num_cach_tasks = self._task_random_uniform_generation(task_num=num_node_tasks,
                                                                                   comp_low=6, comp_high=10,
                                                                                   cach_low=3, cach_high=5)
            num_comp_tasks = num_node_tasks - num_cach_tasks
            # self._task_random_normal_generation
            # 每个节点运行（或者存储）一定数量的计算/缓存任务，并且为确保QoS和系统运行的稳定与高效，考虑每个节点的资源利用率设定一个上限u_lim（i.e.,0.8,0.9,1），
            # 超过该上限值就需要迁移一部分任务到其他的节点中运行，因此，任务

            # 计算出节点中已使用的计算资源和缓存资源
            used_comp = 0
            used_cach = 0
            for i_task in range(len(tasks_info_list)):
                used_comp = used_comp + tasks_info_list[i_task]['dem_comp']
                used_cach = used_cach + tasks_info_list[i_task]['dem_cach']

            # 判断节点中计算/缓存资源是否超出总量，若超出则弹出tasks_info_list末尾的任务（这里认为list右端为末尾，且新任务的task_info从末尾压入该list）
            # 计算资源满，则弹出仅计算任务；存储资源满，则依次弹出计算/缓存任务
            while used_comp > comp_capacity or used_cach > cach_capacity:
                if len(tasks_info_list) == 0:
                    break
                if used_comp > comp_capacity:  # 计算资源满，则仅弹出计算任务
                    for i_rt in range(len(tasks_info_list)):  # 从tasks_info_list找出排的最靠后的计算任务，并弹出它
                        if tasks_info_list[-(i_rt + 1)]['type'] == 'comp':
                            # 更新出节点中已使用的计算资源和缓存资源
                            mig_task = tasks_info_list[-(i_rt + 1)]  # 被（迁移）弹出的task
                            used_comp = used_comp - mig_task['dem_comp']
                            used_cach = used_cach - mig_task['dem_cach']
                            tasks_info_list = tasks_info_list[:-(i_rt + 1)] + tasks_info_list[
                                                                              -(i_rt + 1) + 1:] if i_rt != 0 \
                                else tasks_info_list[:-1]  # 弹出一个task
                            num_node_tasks -= 1
                            num_comp_tasks -= 1
                            break
                elif used_cach > cach_capacity:  # 缓存资源满，则弹出list最末尾的一个计算或者缓存任务
                    # 更新出节点中已使用的计算资源和缓存资源
                    mig_task = tasks_info_list[-1]  # 被（迁移）弹出的task
                    used_comp = used_comp - mig_task['dem_comp']
                    used_cach = used_cach - mig_task['dem_cach']
                    tasks_info_list = tasks_info_list[:-1]
                    num_node_tasks -= 1
                    if mig_task['type'] == 'comp':
                        num_comp_tasks -= 1
                    elif mig_task['type'] == 'cach':
                        num_cach_tasks -= 1
                else:
                    pass
                if used_comp <= comp_capacity and used_cach <= cach_capacity:
                    break

            # num_node_tasks = len(tasks_info_list)
            assert len(
                tasks_info_list) == num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            g_net.DG.nodes[i]['num_node_tasks'] = num_node_tasks  # 节点的任务数量
            g_net.DG.nodes[i]['num_comp_tasks'] = num_comp_tasks  # 节点的计算任务数量
            g_net.DG.nodes[i]['num_cach_tasks'] = num_cach_tasks  # 节点的缓存任务数量
            g_net.DG.nodes[i]['tasks_info_list'] = tasks_info_list  # 更新节点的tasks_info_list
            g_net.DG.nodes[i]['used_comp'] = used_comp  # 更新节点中已使用的计算资源
            g_net.DG.nodes[i]['used_cach'] = used_cach  # 更新节点中已使用的存储资源

        # 链路设置
        for i in range(g_net.num_links):
            s, d = g_net.link_idx_to_sd[i]
            g_net.link_capacities = bandwidth  # Gbps #重新修改初始的链路容量参数
            g_net.DG[s][d]['bandwidth'] = bandwidth  # Gbps
            g_net.DG[s][d]['original_bandwidth'] = bandwidth  # Gbps
            g_net.DG[s][d]['used_bandw'] = 0  # Gbps
        # 假定链路可以专门为一个任务迁移的数据流传输分配其总带宽的 1/32

        return g_net

    def _task_random_uniform_generation(self, task_num, comp_low, comp_high, cach_low, cach_high):
        type = np.random.choice([0, 1], task_num, replace=True, p=[0.7, 0.3])  # 任务类型随机生成，0为计算任务，1为缓存任务;p为被选中的概率
        # dem_comp =  np.random.uniform(comp_low, comp_high, task_num)
        # dem_cach = np.random.uniform(cach_low, cach_high, task_num)
        tasks_info_list = []
        for i in range(task_num):
            if type[i] == 0:
                tasks_info_list.append({'type': 'comp',
                                        'dem_comp': np.random.uniform(comp_low, comp_high),
                                        'dem_cach': 0.5 * np.random.uniform(cach_low, cach_high)})
            else:
                tasks_info_list.append({'type': 'cach',
                                        'dem_comp': 0,
                                        'dem_cach': np.random.uniform(cach_low, cach_high)})
        return tasks_info_list, type.sum().item()

    def _task_random_normal_generation(self, task_num, comp_low, comp_high, cach_low, cach_high):
        type = np.random.choice([0, 1], task_num, replace=True, p=[0.7, 0.3])  # 任务类型随机生成，0为计算任务，1为缓存任务;p为被选中的概率
        # dem_comp =  np.random.uniform(comp_low, comp_high, task_num)
        # dem_cach = np.random.uniform(cach_low, cach_high, task_num)
        tasks_info_list = []
        for i in range(task_num):
            if type[i] == 0:
                tasks_info_list.append({'type': 'comp',
                                        'dem_comp': np.clip(np.random.normal(loc=(comp_high + comp_low) / 2,
                                                                             scale=(comp_high - comp_low) / 2),
                                                            comp_low, comp_high),
                                        'dem_cach': 0.5 * np.clip(np.random.normal(loc=(cach_high + cach_low) / 2,
                                                                                   scale=(cach_high - cach_low) / 2),
                                                                  cach_low, cach_high)})
            else:
                tasks_info_list.append({'type': 'cach',
                                        'dem_comp': 0,
                                        'dem_cach': np.clip(np.random.normal(loc=(cach_high + cach_low) / 2,
                                                                             scale=(cach_high - cach_low) / 2),
                                                            cach_low, cach_high)})
        return tasks_info_list

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

    def _get_g_idx_unassigned_resoure(self):
        g_idx_res_p_comp = {}
        g_idx_res_p_cach = {}
        g_idx_res_p_bandw = {}
        for idx_node in range(self.g_net.num_nodes):
            ### 判断物理网络中剩余未分配的资源是否满足 self.g_net 的需求
            id_node = self.g_net.node_id_list[idx_node]
            idx_p_net = self.node_id_to_idx[id_node]
            g_idx_res_p_comp[idx_node] = self.DG.nodes[idx_p_net]['comp_capacity'] - self.DG.nodes[idx_p_net][
                'allocated_comp']
            g_idx_res_p_cach[idx_node] = self.DG.nodes[idx_p_net]['cach_capacity'] - self.DG.nodes[idx_p_net][
                'allocated_cach']
        for idx_link in range(self.g_net.num_links):
            s, d = self.g_net.link_idx_to_sd[idx_link]
            id_s = self.g_net.node_id_list[s]
            id_d = self.g_net.node_id_list[d]
            idx_s_p_net = self.node_id_to_idx[id_s]
            idx_d_p_net = self.node_id_to_idx[id_d]
            g_idx_res_p_bandw[idx_link] = self.DG[idx_s_p_net][idx_d_p_net]['bandwidth'] \
                                          - self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw']
        unassigned_resoure = (g_idx_res_p_comp, g_idx_res_p_cach, g_idx_res_p_bandw)

        return unassigned_resoure

    def _get_cd_k_to_g_node_link(self, k_net, g_net, assigned_comp_k, assigned_cach_k, assigned_bandw_k):
        # K1_net对于g_net中每个节点的贡献度
        cd_k_to_g_node_comp = np.zeros(g_net.num_nodes)
        cd_k_to_g_node_cach = np.zeros(g_net.num_nodes)
        for idx_node in range(g_net.num_nodes):
            id_node = g_net.node_id_list[idx_node]
            # idx_node_p = self.node_id_to_idx[id_node]
            if id_node in k_net.node_id_list:
                idx_node_k = k_net.node_id_to_idx[id_node]
                if idx_node_k in assigned_comp_k:
                    cd_k_to_g_node_comp[idx_node] = assigned_comp_k[idx_node_k] / g_net.DG.nodes[idx_node][
                        'comp_capacity']
                    cd_k_to_g_node_cach[idx_node] = assigned_cach_k[idx_node_k] / g_net.DG.nodes[idx_node][
                        'cach_capacity']
                else:
                    cd_k_to_g_node_comp[idx_node] = 0
                    cd_k_to_g_node_cach[idx_node] = 0
            else:
                cd_k_to_g_node_comp[idx_node] = 0
                cd_k_to_g_node_cach[idx_node] = 0
        # cd_k_to_g_node_comp = cd_k_to_g_node_comp[np.newaxis, :]
        # cd_k_to_g_node_cach = cd_k_to_g_node_cach[np.newaxis, :]

        cd_k_to_g_link = np.zeros(g_net.num_links)
        for idx_link in range(g_net.num_links):
            s, d = g_net.link_idx_to_sd[idx_link]
            id_s = g_net.node_id_list[s]
            id_d = g_net.node_id_list[d]
            # idx_s_p_net = self.node_id_to_idx[id_s]
            # idx_d_p_net = self.node_id_to_idx[id_d]
            if (id_s, id_d) in k_net.link_id_list:  # !!注意 link_id_list 是乱序的
                idx_s_k = k_net.node_id_to_idx[id_s]
                idx_d_k = k_net.node_id_to_idx[id_d]
                idx_link_k = k_net.link_sd_to_idx[(idx_s_k, idx_d_k)]
                if idx_link_k in assigned_bandw_k:
                    cd_k_to_g_link[idx_link] = assigned_bandw_k[idx_link_k] / g_net.DG[s][d]['bandwidth']
                else:
                    cd_k_to_g_link[idx_link] = 0
            else:
                cd_k_to_g_link[idx_link] = 0
            # cd_k_to_g_link = cd_k_to_g_link[np.newaxis, :]

        return cd_k_to_g_node_comp[np.newaxis, :], cd_k_to_g_node_cach[np.newaxis, :], cd_k_to_g_link[np.newaxis, :]

    def _get_cd_p_to_g_node_link(self, g_net, unassigned_resoure):
        g_idx_res_p_comp, g_idx_res_p_cach, g_idx_res_p_bandw = unassigned_resoure
        # K1_net对于g_net中每个节点的贡献度
        cd_p_to_g_node_comp = np.zeros(g_net.num_nodes)
        cd_p_to_g_node_cach = np.zeros(g_net.num_nodes)
        for idx_node in range(g_net.num_nodes):
            cd_p_to_g_node_comp[idx_node] = g_idx_res_p_comp[idx_node] / g_net.DG.nodes[idx_node]['comp_capacity']
            cd_p_to_g_node_cach[idx_node] = g_idx_res_p_cach[idx_node] / g_net.DG.nodes[idx_node]['cach_capacity']
        cd_p_to_g_link = np.zeros(g_net.num_links)
        for idx_link in range(g_net.num_links):
            s, d = g_net.link_idx_to_sd[idx_link]
            cd_p_to_g_link[idx_link] = g_idx_res_p_bandw[idx_link] / g_net.DG[s][d]['bandwidth']
        return cd_p_to_g_node_comp[np.newaxis, :], cd_p_to_g_node_cach[np.newaxis, :], cd_p_to_g_link[np.newaxis, :]

    def _get_contribution_degree(self, assigned_resoure, unassigned_resoure):

        assigned_comp_k1, assigned_cach_k1, assigned_bandw_k1 = assigned_resoure['k1_net']
        assigned_comp_k2, assigned_cach_k2, assigned_bandw_k2 = assigned_resoure['k2_net']
        assigned_comp_k3, assigned_cach_k3, assigned_bandw_k3 = assigned_resoure['k3_net']
        # g_idx_res_p_comp, g_idx_res_p_cach, g_idx_res_p_bandw = unassigned_resoure

        cd_k1_to_g_node_comp, cd_k1_to_g_node_cach, cd_k1_to_g_link \
            = self._get_cd_k_to_g_node_link(k_net=self.k1_net, g_net=self.g_net,
                                            assigned_comp_k=assigned_comp_k1, assigned_cach_k=assigned_cach_k1,
                                            assigned_bandw_k=assigned_bandw_k1)
        cd_k2_to_g_node_comp, cd_k2_to_g_node_cach, cd_k2_to_g_link \
            = self._get_cd_k_to_g_node_link(k_net=self.k2_net, g_net=self.g_net,
                                            assigned_comp_k=assigned_comp_k2, assigned_cach_k=assigned_cach_k2,
                                            assigned_bandw_k=assigned_bandw_k2)
        cd_k3_to_g_node_comp, cd_k3_to_g_node_cach, cd_k3_to_g_link \
            = self._get_cd_k_to_g_node_link(k_net=self.k3_net, g_net=self.g_net,
                                            assigned_comp_k=assigned_comp_k3, assigned_cach_k=assigned_cach_k3,
                                            assigned_bandw_k=assigned_bandw_k3)
        cd_p_to_g_node_comp, cd_p_to_g_node_cach, cd_p_to_g_link = \
            self._get_cd_p_to_g_node_link(g_net=self.g_net, unassigned_resoure=unassigned_resoure)

        p_n_w_factor = 0.1  # 1 # 未分配的剩余物理网络资源对于生成新业务网络的贡献度的计算比重

        cd_vstack_comp = np.vstack((cd_k1_to_g_node_comp, cd_k2_to_g_node_comp, cd_k3_to_g_node_comp,
                                    p_n_w_factor * cd_p_to_g_node_comp))  # 堆叠
        cd_vstack_comp_norm = cd_vstack_comp / np.sum(cd_vstack_comp, axis=0)  # 归一化
        cd_n_comp = np.sum(cd_vstack_comp_norm, axis=1)  # 得到节点计算资源方面的贡献度

        cd_vstack_cach = np.vstack((cd_k1_to_g_node_cach, cd_k2_to_g_node_cach, cd_k3_to_g_node_cach,
                                    p_n_w_factor * cd_p_to_g_node_cach))  # 堆叠
        cd_vstack_cach_norm = cd_vstack_cach / np.sum(cd_vstack_cach, axis=0)  # 归一化
        cd_n_cach = np.sum(cd_vstack_cach_norm, axis=1)  # 得到节点缓存资源方面的贡献度

        contribution_degree_node = cd_n_comp + cd_n_cach  # [k1, k2, k3, g(p)]

        p_l_w_factor = 0.1

        cd_vstack_link = np.vstack(
            (cd_k1_to_g_link, cd_k2_to_g_link, cd_k3_to_g_link, p_l_w_factor * cd_p_to_g_link))  # 堆叠
        cd_vstack_link_norm = cd_vstack_link / np.sum(cd_vstack_link, axis=0)  # 归一化
        contribution_degree_link = np.sum(cd_vstack_link_norm, axis=1)  # 得到link带宽资源方面的贡献度

        contribution_degree_node = contribution_degree_node / contribution_degree_node.sum()  # 归一化
        contribution_degree_link = contribution_degree_link / contribution_degree_link.sum()  # 归一化
        lam_n = 0.6  # 节点涉及计算和缓存两种资源，其权重因子应该被设置的相对更大些
        lam_e = 0.4
        contribution_degree = lam_n * contribution_degree_node + lam_e * contribution_degree_link
        contribution_degree = contribution_degree / contribution_degree.sum()  # 归一化  # [k1, k2, k3, g(p)]

        return contribution_degree

    def _get_overlap_node_link(self, k_net, g_net):
        # 找到这两个虚拟业务网络中id重叠的全部节点
        id_overlap_nodes = set(k_net.node_id_list).intersection(set(g_net.node_id_list))
        idx_k_overlap_nodes = []
        for i_over in id_overlap_nodes:
            idx_k_overlap_nodes.append(k_net.node_id_to_idx[i_over])
        # 找到这两个虚拟业务网络中id重叠的全部节点
        id_overlap_links = set(k_net.link_id_list).intersection(set(g_net.link_id_list))
        idx_k_overlap_links = []  # idx标识形式的link
        sd_k_overlap_links = []  # (s, d)标识形式的link
        for id_s, id_d in id_overlap_links:
            sd_link = (k_net.node_id_to_idx[id_s], k_net.node_id_to_idx[id_d])
            sd_k_overlap_links.append(sd_link)
            idx_k_overlap_links.append(k_net.link_sd_to_idx[sd_link])

        return idx_k_overlap_nodes, idx_k_overlap_links, sd_k_overlap_links

    def _get_net_node_demand(self, v_net):
        demand_node = []
        for idx_node in range(v_net.num_nodes):
            tasks_info_list = v_net.DG.nodes[idx_node]['tasks_info_list']
            demand_node_comp = []
            demand_node_cach = []
            for i_task in range(len(tasks_info_list)):
                demand_node_comp.append(tasks_info_list[i_task]['dem_comp'])
                demand_node_cach.append(tasks_info_list[i_task]['dem_cach'])
            demand_node.append(np.array(demand_node_comp).sum() + np.array(demand_node_cach).sum())
        demand_n = np.array(demand_node).sum()

        # demand_link = []
        # for idx_link in range(v_net.num_links):
        #     s, d = v_net.link_idx_to_sd[idx_link]
        #     demand_link.append(v_net.DG[s][d]['used_bandw'])
        # demand_l = np.array(demand_link).sum()

        return demand_n

    def _get_g_adjustment(self, g_net, action_g, assigned_resoure, tm_idx=0):

        D_tasks_mig_list = {}
        ###调整容量
        for idx in range(g_net.num_nodes):
            comp_capacity = g_net.DG.nodes[idx]['comp_capacity']  # 节点的计算资源总容量
            cach_capacity = g_net.DG.nodes[idx]['cach_capacity']  # 节点的存储资源总容量
            # comp_effic = g_net.DG.nodes[idx_o_n]['comp_effic']  # 节点的计算效率
            num_node_tasks = g_net.DG.nodes[idx]['num_node_tasks']  # 节点的任务数量
            tasks_info_list = g_net.DG.nodes[idx]['tasks_info_list']
            num_comp_tasks = g_net.DG.nodes[idx]['num_comp_tasks']  # 节点的计算任务数量
            num_cach_tasks = g_net.DG.nodes[idx]['num_cach_tasks']  # 节点的缓存任务数量
            assert num_node_tasks == len(tasks_info_list), "Error from tasks_info_list or num_node_tasks"
            assert num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            # 读取计算出节点中已使用的计算资源量和缓存资源量
            used_comp = g_net.DG.nodes[idx]['used_comp']
            used_cach = g_net.DG.nodes[idx]['used_cach']
            # assert used_comp <= comp_capacity or used_cach <= cach_capacity, "The error if from used_comp or used_cach."
            assert used_comp >= 0 or comp_capacity >= 0 or used_cach >= 0 or cach_capacity >= 0, "The error if from used_xx or used_xx."

            limit_n_allocate = 0.3  # 来限制节点让出（分配出）资源的上限值
            comp_capacity = limit_n_allocate * comp_capacity + (1 - limit_n_allocate) * comp_capacity * (
                        1 - action_g[0])
            cach_capacity = limit_n_allocate * cach_capacity + (1 - limit_n_allocate) * cach_capacity * (
                        1 - action_g[1])

            ### 判断物理网络中剩余未分配的资源是否满足 g_net 的需求
            id_node = g_net.node_id_list[idx]
            idx_p_net = self.node_id_to_idx[id_node]
            res_p_comp = self.DG.nodes[idx_p_net]['comp_capacity'] - self.DG.nodes[idx_p_net]['allocated_comp']
            res_p_cach = self.DG.nodes[idx_p_net]['cach_capacity'] - self.DG.nodes[idx_p_net]['allocated_cach']

            if res_p_comp >= g_net.DG.nodes[idx]['comp_capacity']:  # 满足计算资源需求
                self.DG.nodes[idx_p_net]['allocated_comp'] += comp_capacity  # 更新物理网络节点中已分配的存储资源
                g_done = True
            else:
                g_done = False
                return g_net, 0, 0, g_done

            if res_p_cach >= g_net.DG.nodes[idx]['cach_capacity']:  # 存储计算资源需求
                self.DG.nodes[idx_p_net]['allocated_cach'] += cach_capacity  # 更新物理网络节点中已分配的存储资源
                g_done = True
            else:
                g_done = False
                return g_net, 0, 0, g_done

            # 判断节点中计算/缓存资源是否超出总量，若超出则弹出(迁移)tasks_info_list末尾的任务（这里认为list右端为末尾，且新任务的task_info从末尾压入该list）
            # 计算资源满，则弹出仅计算任务；存储资源满，则依次弹出计算/缓存任务
            tasks_mig_list = []  # 需要迁移的任务
            while used_comp > comp_capacity or used_cach > cach_capacity:
                if len(tasks_info_list) == 0:
                    break
                if used_comp > comp_capacity:  # 计算资源满，则仅弹出计算任务
                    for i_rt in range(len(tasks_info_list)):  # 从tasks_info_list找出排的最靠后的计算任务，并弹出它
                        if tasks_info_list[-(i_rt + 1)]['type'] == 'comp':
                            # 更新出节点中已使用的计算资源和缓存资源
                            mig_task = tasks_info_list[-(i_rt + 1)]  # 被（迁移）弹出的task
                            tasks_mig_list.append(mig_task)
                            used_comp = used_comp - mig_task['dem_comp']
                            used_cach = used_cach - mig_task['dem_cach']
                            tasks_info_list = tasks_info_list[:-(i_rt + 1)] + tasks_info_list[
                                                                              -(i_rt + 1) + 1:] if i_rt != 0 \
                                else tasks_info_list[:-1]  # 弹出一个task
                            num_node_tasks -= 1
                            num_comp_tasks -= 1
                            break
                elif used_cach > cach_capacity:  # 缓存资源满，则弹出list最末尾的一个计算或者缓存任务
                    # 更新出节点中已使用的计算资源和缓存资源
                    mig_task = tasks_info_list[-1]  # 被（迁移）弹出的task
                    tasks_mig_list.append(mig_task)
                    used_comp = used_comp - mig_task['dem_comp']
                    used_cach = used_cach - mig_task['dem_cach']
                    tasks_info_list = tasks_info_list[:-1]
                    num_node_tasks -= 1
                    if mig_task['type'] == 'comp':
                        num_comp_tasks -= 1
                    elif mig_task['type'] == 'cach':
                        num_cach_tasks -= 1
                else:
                    pass
                if used_comp <= comp_capacity and used_cach <= cach_capacity:
                    break
            D_tasks_mig_list[idx] = tasks_mig_list

            # 更新节点属性
            assert len(
                tasks_info_list) == num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            g_net.DG.nodes[idx]['comp_capacity'] = comp_capacity  # 节点的计算资源总容量
            g_net.DG.nodes[idx]['cach_capacity'] = cach_capacity  # 节点的存储资源总容量
            g_net.DG.nodes[idx]['num_node_tasks'] = num_node_tasks  # 节点的任务数量
            g_net.DG.nodes[idx]['num_comp_tasks'] = num_comp_tasks  # 节点的计算任务数量
            g_net.DG.nodes[idx]['num_cach_tasks'] = num_cach_tasks  # 节点的缓存任务数量
            g_net.DG.nodes[idx]['tasks_info_list'] = tasks_info_list  # 更新节点的tasks_info_list
            g_net.DG.nodes[idx]['used_comp'] = used_comp  # 更新节点中已使用的计算资源
            g_net.DG.nodes[idx]['used_cach'] = used_cach  # 更新节点中已使用的存储资源

        # tm_idx = 0  # !!
        eval_tm = g_net.traffic_matrices[tm_idx] / 1e6  # kpbs->Gbps
        eval_link_loads = np.zeros(g_net.num_links)
        eval_link_utilization = np.zeros(g_net.num_links)

        # （1）调整链路资源
        for idx_link in range(g_net.num_links):
            s, d = g_net.link_idx_to_sd[idx_link]
            old_bandwidth = g_net.DG[s][d]['bandwidth']  # 未调整资源前的link带宽
            weight = g_net.DG[s][d]['weight']

            limit_l_allocate = 0.3  # 来限制节点让出（分配出）资源的上限值
            bandwidth = limit_l_allocate * old_bandwidth + (1 - limit_l_allocate) * old_bandwidth * (1 - action_g[2])
            # 在使用NetworkX计算最短路径时，weight的大小对最短路径的结果具有直接影响。这里所说的weight通常指的是图中各条边的权重，
            # 它可以是距离、时间、成本等任何可以量化的数值。权重的大小决定了在计算最短路径时，算法如何评估不同路径的“成本”。
            # NetworkX 通过”最短加权“来计算路径，因此链路的weight越小，计算最短路径越有优势
            weight = weight * (old_bandwidth / (OBJ_EPSILON + bandwidth))  # 根据实际链路容量更新链路权重
            # 更新链路属性
            g_net.DG[s][d]['weight'] = weight
            g_net.DG[s][d]['bandwidth'] = bandwidth

            # 判断是否满足带宽资源，更新基础物理网络
            id_s = g_net.node_id_list[s]
            id_d = g_net.node_id_list[d]
            idx_s_p_net = self.node_id_to_idx[id_s]
            idx_d_p_net = self.node_id_to_idx[id_d]
            if self.DG[idx_s_p_net][idx_d_p_net]['bandwidth'] - self.DG[idx_s_p_net][idx_d_p_net][
                'allocated_bandw'] >= bandwidth:  # 带宽资源是否满足需求
                self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] += g_net.DG[s][d]['bandwidth']  # 更新物理网络链路中已分配的带宽资源
                g_done = True
            else:
                g_done = False
                return g_net, 0, 0, g_done

        # （2）计算链路负载
        # ->方式1：由于仅使用action_g[2]来控制调整所有链路的容量，因此认为所有链路的weight等比例调整，计算出的流的路径不改变
        for idx_flow in range(g_net.num_pairs):
            s, d = g_net.pair_idx_to_sd[idx_flow]
            if eval_tm[s][d] != 0:
                eval_link_loads = self._ksp_traffic_distribution(g_net, eval_link_loads, eval_tm[s][d], s, d,
                                                                 g_net.shortest_paths_node[idx_flow])
        # # ->方式2
        # shortest_path_num = 1  # 注意如果不是 1, 需要采用nx.shortest_simple_paths来计算路径
        # g_net_new_shortest_paths = {}
        # for idx_flow in range(g_net.num_pairs):
        #     s, d = g_net.pair_idx_to_sd[idx_flow]
        #
        #     if eval_tm[s][d] != 0:
        #         paths_list = []
        #         path = nx.shortest_path(g_net.DG, s, d, weight='weight', method='dijkstra')
        #         paths_list.append(path)
        #         g_net_new_shortest_paths[idx_flow] = paths_list
        #
        #         if g_net_new_shortest_paths[idx_flow] != g_net.shortest_paths_node[idx_flow]:  # 新计算路由与初始路由不同
        #             eval_link_loads = self._ksp_traffic_distribution(g_net, eval_link_loads, eval_tm[s][d], s, d,
        #                                                              g_net_new_shortest_paths[idx_flow])
        #         else:
        #             eval_link_loads = self._ksp_traffic_distribution(g_net, eval_link_loads, eval_tm[s][d], s, d,
        #                                                              g_net.shortest_paths_node[idx_flow])

        # （3）计算链路利用率
        for idx_link in range(g_net.num_links):
            s, d = g_net.link_idx_to_sd[idx_link]
            g_net.DG[s][d]['used_bandw'] = eval_link_loads[idx_link]  # 更新used_bandw
            eval_link_utilization[idx_link] = eval_link_loads[idx_link] / g_net.DG[s][d]['bandwidth']

        eval_max_utilization = np.max(eval_link_utilization)
        if g_done == True:
            if eval_max_utilization <= 0.7:
                g_done = True
            else:
                g_done = False  # 是否生成成功，一般链路的利用率最大不超过70%
        else:
            pass

        if g_done == True:
            demand_l = np.array(eval_link_loads).sum()
            demand_n = self._get_net_node_demand(g_net)
        else:
            demand_n = 0
            demand_l = 0

        return g_net, demand_n, demand_l, g_done

    def _get_k_node_cost(self, k_net, idx_overlap_nodes, action_k_node):

        cost_mig_node_list = []
        assigned_comp = {}
        assigned_cach = {}

        no_mig_used_comp = {}
        no_mig_used_cach = {}
        no_mig_comp_capacity = {}
        no_mig_cach_capacity = {}
        D_tasks_mig_list = {}
        ###调整容量
        i_n = 0
        for idx_o_n in idx_overlap_nodes:
            comp_capacity = k_net.DG.nodes[idx_o_n]['comp_capacity']  # 节点的计算资源总容量
            cach_capacity = k_net.DG.nodes[idx_o_n]['cach_capacity']  # 节点的存储资源总容量
            # comp_effic = k_net.DG.nodes[idx_o_n]['comp_effic']  # 节点的计算效率
            num_node_tasks = k_net.DG.nodes[idx_o_n]['num_node_tasks']  # 节点的任务数量
            tasks_info_list = k_net.DG.nodes[idx_o_n]['tasks_info_list']
            num_comp_tasks = k_net.DG.nodes[idx_o_n]['num_comp_tasks']  # 节点的计算任务数量
            num_cach_tasks = k_net.DG.nodes[idx_o_n]['num_cach_tasks']  # 节点的缓存任务数量
            assert num_node_tasks == len(tasks_info_list), "Error from tasks_info_list or num_node_tasks"
            assert num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            # 读取计算出节点中已使用的计算资源量和缓存资源量
            used_comp = k_net.DG.nodes[idx_o_n]['used_comp']
            used_cach = k_net.DG.nodes[idx_o_n]['used_cach']
            # assert used_comp <= comp_capacity or used_cach <= cach_capacity, "The error if from used_comp or used_cach."
            assert used_comp >= 0 or comp_capacity >= 0 or used_cach >= 0 or cach_capacity >= 0, "The error if from used_xx or used_xx."

            no_mig_used_comp[idx_o_n] = used_comp
            no_mig_used_cach[idx_o_n] = used_cach
            no_mig_comp_capacity[idx_o_n] = comp_capacity
            no_mig_cach_capacity[idx_o_n] = cach_capacity

            assigned_comp[idx_o_n] = comp_capacity
            assigned_cach[idx_o_n] = cach_capacity

            limit_n_allocate = 0.2  # 来限制节点让出（分配出）资源的上限值
            comp_capacity = limit_n_allocate * comp_capacity + (1 - limit_n_allocate) * comp_capacity * (
                        1 - action_k_node[i_n])
            cach_capacity = limit_n_allocate * cach_capacity + (1 - limit_n_allocate) * cach_capacity * (
                        1 - action_k_node[i_n])
            assert comp_capacity >= 0 and cach_capacity >= 0, "The comp_capacity and cach_capacity can not be negative."

            assigned_comp[idx_o_n] -= comp_capacity  # 节点分配出去的计算资源
            assigned_cach[idx_o_n] -= cach_capacity  # 节点分配出去的存储资源

            # 判断节点中计算/缓存资源是否超出总量，若超出则弹出(迁移)tasks_info_list末尾的任务（这里认为list右端为末尾，且新任务的task_info从末尾压入该list）
            # 计算资源满，则弹出仅计算任务；存储资源满，则依次弹出计算/缓存任务
            tasks_mig_list = []  # 需要迁移的任务
            while used_comp > comp_capacity or used_cach > cach_capacity:
                if len(tasks_info_list) == 0:
                    break
                if used_comp > comp_capacity:  # 计算资源满，则仅弹出计算任务
                    for i_rt in range(len(tasks_info_list)):  # 从tasks_info_list找出排的最靠后的计算任务，并弹出它
                        if tasks_info_list[-(i_rt + 1)]['type'] == 'comp':
                            # 更新出节点中已使用的计算资源和缓存资源
                            mig_task = tasks_info_list[-(i_rt + 1)]  # 被（迁移）弹出的task
                            tasks_mig_list.append(mig_task)
                            used_comp = used_comp - mig_task['dem_comp']
                            used_cach = used_cach - mig_task['dem_cach']
                            tasks_info_list = tasks_info_list[:-(i_rt + 1)] + tasks_info_list[
                                                                              -(i_rt + 1) + 1:] if i_rt != 0 \
                                else tasks_info_list[:-1]  # 弹出一个task
                            num_node_tasks -= 1
                            num_comp_tasks -= 1
                            break
                elif used_cach > cach_capacity:  # 缓存资源满，则弹出list最末尾的一个计算或者缓存任务
                    # 更新出节点中已使用的计算资源和缓存资源
                    # try:
                    #     mig_task = tasks_info_list[-1]
                    # except IndexError:
                    #     print("列表为空，无法获取最后一个元素")
                    #     mig_task = None
                    mig_task = tasks_info_list[-1]  # 被（迁移）弹出的task
                    tasks_mig_list.append(mig_task)
                    used_comp = used_comp - mig_task['dem_comp']
                    used_cach = used_cach - mig_task['dem_cach']
                    tasks_info_list = tasks_info_list[:-1]
                    num_node_tasks -= 1
                    if mig_task['type'] == 'comp':
                        num_comp_tasks -= 1
                    elif mig_task['type'] == 'cach':
                        num_cach_tasks -= 1
                else:
                    pass
                if used_comp <= comp_capacity and used_cach <= cach_capacity:
                    break
            D_tasks_mig_list[idx_o_n] = tasks_mig_list

            # assert len(tasks_info_list) == num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            # 更新属性
            assert len(
                tasks_info_list) == num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            k_net.DG.nodes[idx_o_n]['comp_capacity'] = comp_capacity  # 节点的计算资源总容量
            k_net.DG.nodes[idx_o_n]['cach_capacity'] = cach_capacity  # 节点的存储资源总容量
            k_net.DG.nodes[idx_o_n]['num_node_tasks'] = num_node_tasks  # 节点的任务数量
            k_net.DG.nodes[idx_o_n]['num_comp_tasks'] = num_comp_tasks  # 节点的计算任务数量
            k_net.DG.nodes[idx_o_n]['num_cach_tasks'] = num_cach_tasks  # 节点的缓存任务数量
            k_net.DG.nodes[idx_o_n]['tasks_info_list'] = tasks_info_list  # 更新节点的tasks_info_list
            k_net.DG.nodes[idx_o_n]['used_comp'] = used_comp  # 更新节点中已使用的计算资源
            k_net.DG.nodes[idx_o_n]['used_cach'] = used_cach  # 更新节点中已使用的存储资源

            # 更新基础物理网络
            id_node = k_net.node_id_list[idx_o_n]
            idx_p_net = self.node_id_to_idx[id_node]
            self.DG.nodes[idx_p_net]['allocated_comp'] += k_net.DG.nodes[idx_o_n]['comp_capacity'] \
                                                          - no_mig_comp_capacity[idx_o_n]  # 更新物理网络节点中已分配的计算资源
            self.DG.nodes[idx_p_net]['allocated_cach'] += k_net.DG.nodes[idx_o_n]['cach_capacity'] \
                                                          - no_mig_cach_capacity[idx_o_n]  # 更新物理网络节点中已分配的存储资源
            i_n = i_n + 1

        ### 任务迁移 ###
        i_n = 0
        for idx_o_n in idx_overlap_nodes:
            # comp_capacity = k_net.DG.nodes[idx_o_n]['comp_capacity']  # 节点的计算资源总容量
            # cach_capacity = k_net.DG.nodes[idx_o_n]['cach_capacity']  # 节点的存储资源总容量
            comp_effic = k_net.DG.nodes[idx_o_n]['comp_effic']  # 节点的计算效率
            num_node_tasks = k_net.DG.nodes[idx_o_n]['num_node_tasks']  # 节点的任务数量
            tasks_info_list = k_net.DG.nodes[idx_o_n]['tasks_info_list']
            num_comp_tasks = k_net.DG.nodes[idx_o_n]['num_comp_tasks']  # 节点的计算任务数量
            num_cach_tasks = k_net.DG.nodes[idx_o_n]['num_cach_tasks']  # 节点的缓存任务数量
            assert num_node_tasks == len(tasks_info_list), "Error from tasks_info_list or num_node_tasks"
            assert num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            # 读取计算出节点中已使用的计算资源量和缓存资源量
            # used_comp = k_net.DG.nodes[idx_o_n]['used_comp']
            # used_cach = k_net.DG.nodes[idx_o_n]['used_cach']

            # 从tasks_mig_list头部开始迁移
            tasks_mig_list = D_tasks_mig_list[idx_o_n]
            cost_mig_nm_list = []  # 任务迁移的代价
            neighbors = k_net.neighbors_idx[idx_o_n]  # 一阶邻居节点
            neighbors_2_order = k_net.neighbors_idx_2_order[idx_o_n]  # 二阶邻居节点
            for i_t_m in range(len(tasks_mig_list)):
                mig_task = tasks_mig_list[i_t_m]
                neighbors_res_comp = []  # 邻居节点剩余计算资源
                neighbors_res_cach = []  # 邻居节点剩余缓存资源
                for i_d in neighbors:
                    neighbors_res_comp.append(
                        k_net.DG.nodes[i_d]['comp_capacity'] - k_net.DG.nodes[i_d]['used_comp'])
                    neighbors_res_cach.append(
                        k_net.DG.nodes[i_d]['cach_capacity'] - k_net.DG.nodes[i_d]['used_cach'])
                neighbors_2_order_res_comp = []  # 二阶邻居节点剩余计算资源
                neighbors_2_order_res_cach = []  # 二阶居节点剩余缓存资源
                for i_d in neighbors_2_order:
                    neighbors_2_order_res_comp.append(
                        k_net.DG.nodes[i_d]['comp_capacity'] - k_net.DG.nodes[i_d]['used_comp'])
                    neighbors_2_order_res_cach.append(
                        k_net.DG.nodes[i_d]['cach_capacity'] - k_net.DG.nodes[i_d]['used_cach'])

                B_nm = 0
                if mig_task['type'] == 'comp':
                    # 找出待迁移任务mig_task的迁移目的节点
                    d_mig_node = None
                    sorted_indices = np.argsort(np.array(neighbors_res_comp))
                    for i_ngb in range(len(neighbors)):  # 遍历一阶邻居节点
                        # 先从一阶邻居节点中选出剩余计算资源最多的节点（同时也要能够满足缓存资源的需求）
                        max_neighbor = neighbors[sorted_indices[-(i_ngb + 1)]]

                        # # 假定链路可以专门为任务迁移的数据流传输分配其初始总带宽（original_bandwidth）的 1/32
                        B_nm = k_net.DG[idx_o_n][max_neighbor]['original_bandwidth'] / 32  # 两节点任务迁移的路径的需求带宽
                        # 两节点之间路径的剩余带宽，（注意如果多跳情况下，路径的带宽为带宽最小的链路的带宽）
                        B_nm_res = k_net.DG[idx_o_n][max_neighbor]['bandwidth'] - k_net.DG[idx_o_n][max_neighbor][
                            'used_bandw']

                        max_neighbor_res_comp = k_net.DG.nodes[max_neighbor]['comp_capacity'] - \
                                                k_net.DG.nodes[max_neighbor]['used_comp']
                        max_neighbor_res_cach = k_net.DG.nodes[max_neighbor]['cach_capacity'] - \
                                                k_net.DG.nodes[max_neighbor]['used_cach']

                        # 判断该邻居节点是否满足计算和缓存资源的需求, 同时源节点与迁移的目的节点之间的路径带宽满足B_nm的要求
                        if (max_neighbor_res_comp >= mig_task['dem_comp'] and max_neighbor_res_cach >= mig_task[
                            'dem_cach']) \
                                and B_nm_res >= B_nm:
                            d_mig_node = max_neighbor
                            break

                    if d_mig_node == None:  # 当一阶邻居节点都不满足，遍历二阶邻居节点
                        sorted_indices = np.argsort(np.array(neighbors_2_order_res_comp))
                        for i_ngb in range(len(neighbors_2_order)):
                            # 从二阶邻居节点中选出剩余计算资源最多的节点（同时也要能够满足缓存资源的需求）
                            max_neighbor = neighbors_2_order[sorted_indices[-(i_ngb + 1)]]

                            # # 假定链路可以专门为任务迁移的数据流传输分配其初始总带宽（original_bandwidth）的 1/32
                            # B_nm = k_net.DG[idx_o_n][max_neighbor]['original_bandwidth'] / 32  # 两节点任务迁移的路径的需求带宽
                            # 两节点之间路径的剩余带宽，（注意如果多跳情况下，路径的带宽为带宽最小的链路的带宽）
                            paths_nm = k_net.shortest_paths_link[
                                k_net.pair_sd_to_idx[(idx_o_n, max_neighbor)]]
                            B_nm_paths = 0
                            B_nm_res = 0
                            for i_path in paths_nm:
                                B_nm_path = []
                                B_nm_res_path = []
                                for i_link in i_path:
                                    s, d = k_net.link_idx_to_sd[i_link]
                                    B_nm_path.append(k_net.DG[s][d]['bandwidth'])
                                    B_nm_res_path.append(k_net.DG[s][d]['bandwidth'] - k_net.DG[s][d]['used_bandw'])
                                B_nm_paths = B_nm_paths + min(B_nm_path)
                                B_nm_res = B_nm_res + min(B_nm_res_path)
                            B_nm = B_nm_paths / 32  # 两节点任务迁移的路径的需求带宽

                            max_neighbor_res_comp = k_net.DG.nodes[max_neighbor]['comp_capacity'] - \
                                                    k_net.DG.nodes[max_neighbor]['used_comp']
                            max_neighbor_res_cach = k_net.DG.nodes[max_neighbor]['cach_capacity'] - \
                                                    k_net.DG.nodes[max_neighbor]['used_cach']

                            # 判断该邻居节点是否满足计算和缓存资源的需求, 同时源节点与迁移的目的节点之间的路径带宽满足B_nm的要求
                            if (max_neighbor_res_comp >= mig_task['dem_comp'] and max_neighbor_res_cach >= mig_task[
                                'dem_cach']) \
                                    and B_nm_res >= B_nm:
                                d_mig_node = max_neighbor
                                break

                    if d_mig_node == None:  # 当二阶邻居节点仍然都不满足，则遗弃该待迁移的任务，并反馈一个cost
                        # tasks_mig_list = tasks_mig_list[1:]  # 从待迁移任务队列中移出任务
                        cost_mig_nm = 50  # !! 任务遗弃的惩罚代价
                        cost_mig_nm_list.append(cost_mig_nm)
                        continue

                    # 判断该邻居节点是否满足计算和缓存资源的需求, 同时源节点与迁移的目的节点之间的路径带宽满足B_nm的要求
                    k_net.DG.nodes[d_mig_node]['tasks_info_list'] += [mig_task]  # 从待迁移任务队列中迁入一个任务到目的节点
                    k_net.DG.nodes[d_mig_node]['used_comp'] += mig_task['dem_comp']  # 更新邻居节点的资源使用
                    k_net.DG.nodes[d_mig_node]['used_cach'] += mig_task['dem_cach']  # 更新邻居节点的资源使用
                    k_net.DG.nodes[d_mig_node]['num_node_tasks'] += 1  # 更新邻居节点的任务数量
                    k_net.DG.nodes[d_mig_node]['num_comp_tasks'] += 1  # 更新邻居节点的计算任务数量

                    # cost计算
                    # 迁移时延
                    tau_mig = mig_task['dem_cach'] / B_nm  # B_nm!! # 这里我们认为迁移任务所需要的缓存资源mig_task['dem_cach']等于它的数据量大小

                    # 数据流迁移在目的节点中造成的缓冲区队列等待时延, 主要与计算任务缓存区队列长度有关
                    tau_q = (k_net.DG.nodes[d_mig_node]['used_comp'] / k_net.DG.nodes[d_mig_node]['comp_capacity']) \
                            * k_net.DG.nodes[d_mig_node]['num_comp_tasks'] \
                            * np.clip(np.random.normal(loc=1, scale=0.2), 0.8, 1.2)

                    # 数据流在迁移源和目的节点的处理时延, 与节点计算资源利用率和正处理的任务数量等方面有关
                    tau_p_s = 0.5 * comp_effic * \
                              (no_mig_used_comp[idx_o_n] / no_mig_comp_capacity[idx_o_n]) \
                              * k_net.DG.nodes[idx_o_n]['num_comp_tasks'] \
                              * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)

                    # 与迁移数量流的数据量正相关的基础QoS惩罚因子
                    tau_p_d = 0.5 * k_net.DG.nodes[d_mig_node]['comp_effic'] \
                              * (k_net.DG.nodes[d_mig_node]['used_comp'] / k_net.DG.nodes[d_mig_node]['comp_capacity']) \
                              * k_net.DG.nodes[d_mig_node]['num_comp_tasks'] \
                              * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)

                    rho_mig_nm = 0.5 * mig_task['dem_cach'] / 5 + 0.5 * np.clip(np.random.normal(loc=1, scale=0.2), 0.8,
                                                                                1.2)
                    gamma_m_comp = 0.5 * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)
                    gamma_m_cach = 0.5 * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)
                    cost_mig_nm = rho_mig_nm * (tau_mig + tau_q + tau_p_d - tau_p_s) \
                                  + gamma_m_comp * mig_task['dem_comp'] + gamma_m_cach * mig_task['dem_cach']
                    cost_mig_nm_list.append(cost_mig_nm)

                elif mig_task['type'] == 'cach':
                    # 找出待迁移任务mig_task的迁移目的节点
                    d_mig_node = None
                    sorted_indices = np.argsort(np.array(neighbors_res_cach))
                    for i_ngb in range(len(neighbors)):  # 遍历一阶邻居节点
                        # 先从一阶邻居节点中选出剩余存储资源最多的节点（同时也要能够满足缓存资源的需求）
                        max_neighbor = neighbors[sorted_indices[-(i_ngb + 1)]]

                        # # 假定链路可以专门为任务迁移的数据流传输分配其初始总带宽（original_bandwidth）的 1/32
                        B_nm = k_net.DG[idx_o_n][max_neighbor]['original_bandwidth'] / 32  # 两节点任务迁移的路径的需求带宽
                        # 两节点之间路径的剩余带宽，（注意如果多跳情况下，路径的带宽为带宽最小的链路的带宽）
                        B_nm_res = k_net.DG[idx_o_n][max_neighbor]['bandwidth'] - k_net.DG[idx_o_n][max_neighbor][
                            'used_bandw']

                        # max_neighbor_res_comp = k_net.DG.nodes[max_neighbor]['comp_capacity'] - k_net.DG.nodes[max_neighbor]['used_comp']
                        max_neighbor_res_cach = k_net.DG.nodes[max_neighbor]['cach_capacity'] - \
                                                k_net.DG.nodes[max_neighbor]['used_cach']

                        # 判断该邻居节点是否满足计算和缓存资源的需求, 同时源节点与迁移的目的节点之间的路径带宽满足B_nm的要求
                        if max_neighbor_res_cach >= mig_task['dem_cach'] and B_nm_res >= B_nm:
                            d_mig_node = max_neighbor
                            break

                    if d_mig_node == None:  # 当一阶邻居节点都不满足，遍历二阶邻居节点
                        sorted_indices = np.argsort(np.array(neighbors_2_order_res_cach))
                        for i_ngb in range(len(neighbors_2_order)):
                            # 从二阶邻居节点中选出剩余计算资源最多的节点（同时也要能够满足缓存资源的需求）
                            max_neighbor = neighbors_2_order[sorted_indices[-(i_ngb + 1)]]

                            # # 假定链路可以专门为任务迁移的数据流传输分配其初始总带宽（original_bandwidth）的 1/32
                            # B_nm = k_net.DG[idx_o_n][max_neighbor]['original_bandwidth'] / 32  # 两节点任务迁移的路径的需求带宽
                            # 两节点之间路径的剩余带宽，（注意如果多跳情况下，路径的带宽为带宽最小的链路的带宽）
                            paths_nm = k_net.shortest_paths_link[
                                k_net.pair_sd_to_idx[(idx_o_n, max_neighbor)]]
                            B_nm_paths = 0
                            B_nm_res = 0
                            for i_path in paths_nm:
                                B_nm_path = []
                                B_nm_res_path = []
                                for i_link in i_path:
                                    s, d = k_net.link_idx_to_sd[i_link]
                                    B_nm_path.append(k_net.DG[s][d]['bandwidth'])
                                    B_nm_res_path.append(k_net.DG[s][d]['bandwidth'] - k_net.DG[s][d]['used_bandw'])
                                B_nm_paths = B_nm_paths + min(B_nm_path)
                                B_nm_res = B_nm_res + min(B_nm_res_path)
                            B_nm = B_nm_paths / 32  # 两节点任务迁移的路径的需求带宽

                            # max_neighbor_res_comp = k_net.DG.nodes[max_neighbor]['comp_capacity'] - k_net.DG.nodes[max_neighbor]['used_comp']
                            max_neighbor_res_cach = k_net.DG.nodes[max_neighbor]['cach_capacity'] - \
                                                    k_net.DG.nodes[max_neighbor]['used_cach']

                            # 判断该邻居节点是否满足计算和缓存资源的需求, 同时源节点与迁移的目的节点之间的路径带宽满足B_nm的要求
                            if max_neighbor_res_cach >= mig_task['dem_cach'] and B_nm_res >= B_nm:
                                d_mig_node = max_neighbor
                                break

                    if d_mig_node == None:  # 当二阶邻居节点仍然都不满足，则遗弃该待迁移的任务，并反馈一个cost
                        # tasks_mig_list = tasks_mig_list[1:]  # 从待迁移任务队列中移出任务
                        cost_mig_nm = 50  # !! 任务遗弃的惩罚代价
                        cost_mig_nm_list.append(cost_mig_nm)
                        continue

                    # 判断该邻居节点是否满足计算和缓存资源的需求, 同时源节点与迁移的目的节点之间的路径带宽满足B_nm的要求
                    k_net.DG.nodes[d_mig_node]['tasks_info_list'] += [mig_task]  # 从待迁移任务队列中迁入一个任务到目的节点
                    k_net.DG.nodes[d_mig_node]['used_comp'] += mig_task['dem_comp']  # 更新邻居节点的资源使用
                    k_net.DG.nodes[d_mig_node]['used_cach'] += mig_task['dem_cach']  # 更新邻居节点的资源使用
                    k_net.DG.nodes[d_mig_node]['num_node_tasks'] += 1  # 更新邻居节点的任务数量
                    k_net.DG.nodes[d_mig_node]['num_cach_tasks'] += 1  # 更新邻居节点的缓存任务数量
                    # cost计算
                    # 迁移时延
                    tau_mig = mig_task['dem_cach'] / B_nm  # B_nm!! # 这里我们认为迁移任务所需要的缓存资源mig_task['dem_cach']等于它的数据量大小

                    # 数据流迁移在目的节点中造成的缓冲区队列等待时延, 主要与存储任务的缓存区队列长度有关
                    tau_q = (k_net.DG.nodes[d_mig_node]['used_cach'] / k_net.DG.nodes[d_mig_node]['cach_capacity']) \
                            * k_net.DG.nodes[d_mig_node]['num_cach_tasks'] \
                            * np.clip(np.random.normal(loc=1, scale=0.2), 0.8, 1.2)

                    # 数据流在迁移源和目的节点的处理时延, 与节点资源利用率和正处理的任务数量等方面有关
                    tau_p_s = 0.2 * (no_mig_used_cach[idx_o_n] / no_mig_cach_capacity[idx_o_n]) \
                              * k_net.DG.nodes[idx_o_n]['num_cach_tasks'] \
                              * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)

                    # 与迁移数量流的数据量正相关的基础QoS惩罚因子
                    tau_p_d = 0.2 * (
                                k_net.DG.nodes[d_mig_node]['used_cach'] / k_net.DG.nodes[d_mig_node]['cach_capacity']) \
                              * k_net.DG.nodes[d_mig_node]['num_cach_tasks'] \
                              * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)

                    rho_mig_nm = 0.5 * mig_task['dem_cach'] / 5 + 0.5 * np.clip(np.random.normal(loc=1, scale=0.2), 0.8,
                                                                                1.2)
                    gamma_m_comp = 0.5 * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)
                    gamma_m_cach = 0.5 * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)
                    cost_mig_nm = rho_mig_nm * (tau_mig + tau_q + tau_p_d - tau_p_s) \
                                  + gamma_m_comp * mig_task['dem_comp'] + gamma_m_cach * mig_task['dem_cach']
                    cost_mig_nm_list.append(cost_mig_nm)

            cost_mig_node = np.array(cost_mig_nm_list).sum()  # 一个节点内任务迁移所带来的代价
            cost_mig_node_list.append(cost_mig_node)

            # # num_node_tasks = len(tasks_info_list)
            # assert len(tasks_info_list) == num_node_tasks == num_comp_tasks + num_cach_tasks, "Error from the number of tasks"
            # k_net.DG.nodes[idx_o_n]['num_node_tasks'] = num_node_tasks  # 节点的任务数量
            # k_net.DG.nodes[idx_o_n]['num_comp_tasks'] = num_comp_tasks  # 节点的计算任务数量
            # k_net.DG.nodes[idx_o_n]['num_cach_tasks'] = num_cach_tasks  # 节点的缓存任务数量
            # k_net.DG.nodes[idx_o_n]['tasks_info_list'] = tasks_info_list  # 更新节点的tasks_info_list
            # k_net.DG.nodes[idx_o_n]['used_comp'] = used_comp  # 更新节点中已使用的计算资源
            # k_net.DG.nodes[idx_o_n]['used_cach'] = used_cach  # 更新节点中已使用的存储资源

            i_n = i_n + 1

        cost_n = np.array(cost_mig_node_list).sum()  # 业务网络中让出多个节点的资源所带来的代价
        return cost_n, k_net, assigned_comp, assigned_cach

    def _get_k_link_cost(self, k_net, idx_overlap_links, action_k_link, tm_idx):
        # tm_idx = 0  # !!
        eval_tm = k_net.traffic_matrices[tm_idx] / 1e6  # kpbs->Gbps
        eval_link_loads = np.zeros(k_net.num_links)
        eval_link_utilization = np.zeros(k_net.num_links)
        old_link_loads = np.zeros(k_net.num_links)
        old_link_utilization = np.zeros(k_net.num_links)

        traffic_reroute = 0
        flow_reroute = 0
        # （0）计算链路权重更改前的链路负载
        for idx_flow in range(k_net.num_pairs):
            s, d = k_net.pair_idx_to_sd[idx_flow]
            if eval_tm[s][d] != 0:
                old_link_loads = self._ksp_traffic_distribution(k_net, old_link_loads, eval_tm[s][d], s, d,
                                                                k_net.shortest_paths_node[idx_flow])
        old_bandwidth = np.zeros(k_net.num_links)
        for idx_link in range(k_net.num_links):
            s, d = k_net.link_idx_to_sd[idx_link]
            old_bandwidth[idx_link] = k_net.DG[s][d]['bandwidth']  # 未让出资源前的link带宽
            old_link_utilization[idx_link] = old_link_loads[idx_link] / k_net.DG[s][d]['bandwidth']
        old_max_utilization = np.max(old_link_utilization)

        # （1）更改链路容量和权重
        i_l = 0
        assigned_bandw = {}
        for idx_o_l in idx_overlap_links:
            s, d = k_net.link_idx_to_sd[idx_o_l]

            link_capacities = k_net.link_capacities  # 该变量暂不使用 # Gbps
            bandwidth = k_net.DG[s][d]['bandwidth']
            original_bandwidth = k_net.DG[s][d]['original_bandwidth']
            used_bandw = k_net.DG[s][d]['used_bandw']
            weight = k_net.DG[s][d]['weight']

            # old_bandwidth[idx_o_l] = bandwidth  # 未让出资源前的link带宽
            limit_l_allocate = 0.2  # 来限制节点让出（分配出）资源的上限值
            bandwidth = limit_l_allocate * bandwidth + (1 - limit_l_allocate) * bandwidth * (1 - action_k_link[i_l])
            # 在使用NetworkX计算最短路径时，weight的大小对最短路径的结果具有直接影响。这里所说的weight通常指的是图中各条边的权重，
            # 它可以是距离、时间、成本等任何可以量化的数值。权重的大小决定了在计算最短路径时，算法如何评估不同路径的“成本”。
            # NetworkX 通过”最短加权“来计算路径，因此链路的weight越小，计算最短路径越有优势
            weight = weight * (old_bandwidth[idx_o_l] / (OBJ_EPSILON + bandwidth))  # 根据实际链路容量更新链路权重
            assert bandwidth >= 0 and weight >= 0, "The bandwidth and weight can not be negative."
            # 更新属性
            k_net.DG[s][d]['weight'] = weight
            k_net.DG[s][d]['bandwidth'] = bandwidth

            assigned_bandw[idx_o_l] = old_bandwidth[idx_o_l] - bandwidth

            # 更新基础物理网络
            id_s = k_net.node_id_list[s]
            id_d = k_net.node_id_list[d]
            idx_s_p_net = self.node_id_to_idx[id_s]
            idx_d_p_net = self.node_id_to_idx[id_d]
            self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] += k_net.DG[s][d]['bandwidth'] \
                                                                    - old_bandwidth[idx_o_l]  # 物理网络链路中已分配的带宽资源
            i_l += 1

        # （2）重新计算数据流的路径
        shortest_path_num = 1  # 注意如果不是 1, 需要采用nx.shortest_simple_paths来计算路径
        k_net_new_shortest_paths = {}
        for idx_flow in range(k_net.num_pairs):
            s, d = k_net.pair_idx_to_sd[idx_flow]

            if eval_tm[s][d] != 0:
                paths_list = []
                path = nx.shortest_path(k_net.DG, s, d, weight='weight', method='dijkstra')
                paths_list.append(path)
                k_net_new_shortest_paths[idx_flow] = paths_list

                if k_net_new_shortest_paths[idx_flow] != k_net.shortest_paths_node[idx_flow]:  # 新计算路由与初始路由不同
                    traffic_reroute += eval_tm[s][d]  ##**#
                    flow_reroute += 1  ##**#
                    eval_link_loads = self._ksp_traffic_distribution(k_net, eval_link_loads, eval_tm[s][d], s, d,
                                                                     k_net_new_shortest_paths[idx_flow])
                else:
                    eval_link_loads = self._ksp_traffic_distribution(k_net, eval_link_loads, eval_tm[s][d], s, d,
                                                                     k_net.shortest_paths_node[idx_flow])

        for idx_link in range(k_net.num_links):
            s, d = k_net.link_idx_to_sd[idx_link]
            k_net.DG[s][d]['used_bandw'] = eval_link_loads[idx_link]  # 更新used_bandw
            eval_link_utilization[idx_link] = eval_link_loads[idx_link] / k_net.DG[s][d]['bandwidth']
        eval_max_utilization = np.max(eval_link_utilization)

        # cost 计算
        # 收敛代价 # 网络收敛受网络规模、网络拓扑结构、路由器性能、网络负载和带宽等多方面因素的影响
        rho_cvg = 0.3 * (flow_reroute / k_net.num_pairs) \
                  + 0.5 * np.log(k_net.num_pairs) / np.log(self.num_pairs) \
                  + 0.2 * np.clip(np.random.normal(loc=1, scale=0.1), 0.8, 1.2)
        router_performance_index = 0
        for i in range(k_net.num_nodes):
            router_performance_index += k_net.DG.nodes[i]['router_performance'] / k_net.num_nodes
        bandw_performance_index = 0
        res_bandw_list = []
        bandwidth_list = []
        for i in range(k_net.num_links):
            s, d = k_net.link_idx_to_sd[i]
            res_bandw_list.append(k_net.DG[s][d]['bandwidth'] - k_net.DG[s][d]['used_bandw'])
            bandwidth_list.append(k_net.DG[s][d]['bandwidth'])
            bandw_performance_index += (k_net.DG[s][d]['bandwidth'] / (
                    OBJ_EPSILON + k_net.DG[s][d]['bandwidth'] - k_net.DG[s][d]['used_bandw'])) \
                                       * (old_bandwidth[i] / (OBJ_EPSILON + k_net.DG[s][d]['bandwidth'])) \
                                       / k_net.num_links

        tau_cvg = np.log(k_net.num_nodes) * np.log(k_net.num_links) \
                  * router_performance_index * bandw_performance_index \
                  * 10 * eval_max_utilization * np.sum(eval_link_utilization) \
                  * (old_bandwidth.sum() / np.array(res_bandw_list).sum())

        cost_cvg = rho_cvg * tau_cvg  # 业务网络的网络收敛的代价

        rho_RD = 10
        cost_RD = rho_RD * traffic_reroute / eval_tm.sum()  # 流的重路由干扰的代价

        rho_LB = 500 * np.sum(eval_link_utilization) / len(eval_link_utilization)
        cost_LB = rho_LB * (eval_max_utilization - old_max_utilization) / old_max_utilization  # 业务网络的链路负载均衡的代价

        cost_l = cost_cvg + cost_RD + cost_LB

        return cost_l, k_net, assigned_bandw, np.array(eval_link_loads).sum(), np.array(old_link_loads).sum()

    def _get_v_net_observations(self, v_net, idx_k_overlap_nodes, idx_k_overlap_links, is_zero_padding):
        # 节点信息[是否节点重叠, comp资源容量（比）, comp资源利用率, cach资源容量（比）, cach资源利用率]
        node_obs_list = []
        for idx_node in range(v_net.num_nodes):
            id_node = v_net.node_id_list[idx_node]
            idx_p_net = self.node_id_to_idx[id_node]

            comp_capacity = v_net.DG.nodes[idx_node]['comp_capacity']
            used_comp = v_net.DG.nodes[idx_node]['used_comp']
            p_comp_capacity = self.DG.nodes[idx_p_net]['comp_capacity']

            cach_capacity = v_net.DG.nodes[idx_node]['cach_capacity']
            used_cach = v_net.DG.nodes[idx_node]['used_cach']
            p_cach_capacity = self.DG.nodes[idx_p_net]['cach_capacity']

            overlap = 1 if idx_node in idx_k_overlap_nodes else 0

            node_obs_list += [overlap,
                              comp_capacity / p_comp_capacity, used_comp / comp_capacity,
                              cach_capacity / p_cach_capacity, used_cach / cach_capacity]

        # 链路信息[是否链路重叠, bandw资源容量（比）, bandw资源利用率]
        link_obs_list = []
        for idx_link in range(v_net.num_links):
            s, d = v_net.link_idx_to_sd[idx_link]
            bandwidth = v_net.DG[s][d]['bandwidth']  # Gbps
            used_bandw = v_net.DG[s][d]['used_bandw']  # Gbps
            id_s = v_net.node_id_list[s]
            id_d = v_net.node_id_list[d]
            idx_s_p_net = self.node_id_to_idx[id_s]
            idx_d_p_net = self.node_id_to_idx[id_d]
            p_bandwidth = self.DG[idx_s_p_net][idx_d_p_net]['bandwidth']
            # 检查
            if self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] > self.DG[idx_s_p_net][idx_d_p_net]['bandwidth']:
                print("Error")
            # assert self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] <= self.DG[idx_s_p_net][idx_d_p_net]['bandwidth'], \
            #     "Error from the allocated_bandw."

            overlap = 1 if idx_link in idx_k_overlap_links else 0

            link_obs_list += [overlap, bandwidth / p_bandwidth, used_bandw / bandwidth]

        #  网络拓扑信息（邻接矩阵）
        adj_matrix = np.zeros((v_net.num_nodes, v_net.num_nodes), dtype=np.float32)
        adj_coo = np.zeros((2, v_net.num_links), dtype=np.float32)
        adj_link = []
        for idx, (node1, node2) in v_net.link_idx_to_sd.items():
            adj_matrix[node1, node2] = 1
            adj_coo[0, idx] = node1
            adj_coo[1, idx] = node2
            adj_link += [node1, node2]
        # # 方式一，使用adj_matrix，并直接展开成向量
        # adj_obs = adj_matrix.reshape(-1)
        #
        # # 方式二， 使用adj_coo， 并直接展开成向量
        # adj_obs = adj_coo.reshape(-1)
        # adj_obs = adj_obs / (np.max(adj_obs) - np.min(adj_obs))  # 归一化

        # 方式三，根据adj_coo按照Link信息展开成向量
        adj_obs = np.array(adj_link, dtype=np.float32)
        adj_obs = adj_obs / (np.max(adj_obs) - np.min(adj_obs))  # 归一化

        ## 补零
        if is_zero_padding == True:
            link_padding = []
            adj_link_padding = []
            for i_p in range(self.max_num_links - v_net.num_links):
                link_padding += [0, 0, 0]  # [0.5, 0.5, 0.5]  # [0, 0, 0]
                adj_link_padding += [0, 0]  # [0.5, 0.5]  # [0, 0]

            obs_k = np.array(node_obs_list + link_obs_list + link_padding
                             + list(adj_obs) + adj_link_padding, dtype=np.float32)  # 注意 obs的取值范围为[0, 1]
        else:
            obs_k = np.array(node_obs_list + link_obs_list + list(adj_obs), dtype=np.float32)  # 注意 obs的取值范围为[0, 1]，
        # # 检查 数组中是否存在NaN
        # if np.isnan(obs_k).any() == True:
        #     pass
        if np.any(obs_k > 1) == True:
            print("Error")
        elif np.any(obs_k < 0) == True:
            print("Error")

        obs_k = np.clip(obs_k, 0, 1)

        return obs_k

    def _get_g_net_observations(self, is_zero_padding):
        # 节点信息[物理网络剩余comp资源（比）, 预分配（请求）comp资源, 物理网络剩余cach资源（比）, 预分配（请求）cach资源]
        node_obs_list = []
        for idx_node in range(self.g_net.num_nodes):
            id_node = self.g_net.node_id_list[idx_node]
            idx_p_net = self.node_id_to_idx[id_node]

            comp_capacity = self.g_net.DG.nodes[idx_node]['comp_capacity']
            # used_comp = self.g_net.DG.nodes[idx_node]['used_comp']
            p_comp_capacity = self.DG.nodes[idx_p_net]['comp_capacity']
            p_allocated_comp = self.DG.nodes[idx_p_net]['allocated_comp']

            cach_capacity = self.g_net.DG.nodes[idx_node]['cach_capacity']
            # used_cach = self.g_net.DG.nodes[idx_node]['used_cach']
            p_cach_capacity = self.DG.nodes[idx_p_net]['cach_capacity']
            p_allocated_cach = self.DG.nodes[idx_p_net]['allocated_cach']

            if is_zero_padding == True:
                node_obs_list += [0,
                                  p_allocated_comp / p_comp_capacity, comp_capacity / p_comp_capacity,
                                  p_allocated_cach / p_cach_capacity, cach_capacity / p_cach_capacity]
            else:
                node_obs_list += [p_allocated_comp / p_comp_capacity, comp_capacity / p_comp_capacity,
                                  p_allocated_cach / p_cach_capacity, cach_capacity / p_cach_capacity]

        # 链路信息[物理网络剩余bandw资源（比）, 预分配（请求）bandw资源]
        link_obs_list = []
        for idx_link in range(self.g_net.num_links):
            s, d = self.g_net.link_idx_to_sd[idx_link]
            bandwidth = self.g_net.DG[s][d]['bandwidth']  # Gbps
            # used_bandw = self.g_net.DG[s][d]['used_bandw']  # Gbps
            id_s = self.g_net.node_id_list[s]
            id_d = self.g_net.node_id_list[d]
            idx_s_p_net = self.node_id_to_idx[id_s]
            idx_d_p_net = self.node_id_to_idx[id_d]
            p_bandwidth = self.DG[idx_s_p_net][idx_d_p_net]['bandwidth']
            p_allocated_bandw = self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw']
            # 检查
            if self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] > self.DG[idx_s_p_net][idx_d_p_net]['bandwidth']:
                print("Error")
            # assert self.DG[idx_s_p_net][idx_d_p_net]['allocated_bandw'] <= self.DG[idx_s_p_net][idx_d_p_net]['bandwidth'], \
            #     "Error from the allocated_bandw."

            if is_zero_padding == True:
                link_obs_list += [0,
                                  p_allocated_bandw / p_bandwidth, bandwidth / p_bandwidth]
            else:
                link_obs_list += [p_allocated_bandw / p_bandwidth, bandwidth / p_bandwidth]

        # 网络拓扑信息（邻接矩阵）
        adj_matrix = np.zeros((self.g_net.num_nodes, self.g_net.num_nodes), dtype=np.float32)
        adj_coo = np.zeros((2, self.g_net.num_links), dtype=np.float32)
        adj_link = []
        for idx, (node1, node2) in self.g_net.link_idx_to_sd.items():
            adj_matrix[node1, node2] = 1
            adj_coo[0, idx] = node1
            adj_coo[1, idx] = node2
            adj_link += [node1, node2]
        # # 方式一，使用adj_matrix，并直接展开成向量
        # adj_obs = adj_matrix.reshape(-1)
        #
        # # 方式二， 使用adj_coo， 并直接展开成向量
        # adj_obs = adj_coo.reshape(-1)
        # adj_obs = adj_obs / (np.max(adj_obs) - np.min(adj_obs))  # 归一化

        # 方式三，根据adj_coo按照Link信息展开成向量
        adj_obs = np.array(adj_link, dtype=np.float32)
        adj_obs = adj_obs / (np.max(adj_obs) - np.min(adj_obs))  # 归一化

        ## 补零
        if is_zero_padding == True:
            link_padding = []
            adj_link_padding = []
            for i_p in range(self.max_num_links - self.g_net.num_links):
                link_padding += [0, 0, 0]  # [0.5, 0.5, 0.5]  # [0, 0, 0]
                adj_link_padding += [0, 0]  # [0.5, 0.5]  # [0, 0]

            obs_g = np.array(node_obs_list + link_obs_list + link_padding
                             + list(adj_obs) + adj_link_padding, dtype=np.float32)  # 注意 obs的取值范围为[0, 1]
        else:
            obs_g = np.array(node_obs_list + link_obs_list + list(adj_obs), dtype=np.float32)  # 注意 obs的取值范围为[0, 1]，

        # 检查
        if np.any(obs_g > 1) == True:
            print("Error")
        elif np.any(obs_g < 0) == True:
            print("Error")

        obs_g = np.clip(obs_g, 0, 1)
        return obs_g

    def _get_scale_repeat_action(self, action_k_n_l, idx_k_overlap):

        if len(action_k_n_l) < len(idx_k_overlap):
            # assert len(idx_k_overlap) <= len(action_k_n_l) * control_scale_n\
            #        and len(idx_k_overlap) > (len(action_k_n_l)-1) * control_scale_n, "Error from idx_k_overlap or action_k1_node "
            control_scale = math.ceil(len(idx_k_overlap) / len(action_k_n_l))
            action_k_n_l = np.repeat(action_k_n_l, control_scale)[:len(idx_k_overlap)]
        else:
            control_scale = 1
            action_k_n_l = action_k_n_l[:len(idx_k_overlap)]

        return action_k_n_l, control_scale

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

    def _eval_ecmp_traffic_distribution(self, tm_idx, eval_delay=False):
        eval_link_loads = self._ecmp_traffic_distribution(tm_idx)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay, eval_link_loads / self.link_capacities

    def _ksp_traffic_distribution(self, v_net, link_loads, demand, src, dst, paths):
        if src == dst:
            return
        traffic_demand = demand / len(paths) * np.ones(len(paths))
        for i in range(len(paths)):
            for j in range(len(paths[i]) - 1):
                link_loads[v_net.link_sd_to_idx[(paths[i][j], paths[i][j + 1])]] += traffic_demand[i]

        return link_loads


class InitVnetworkTrain(object):

    def __init__(self, config, data_dir, v_DG, str_v_net_name, v_remove_nodes_idx):

        self.data_dir = data_dir
        self.v_DG = v_DG

        self.v_num_nodes = self.v_DG.number_of_nodes() - len(v_remove_nodes_idx)  # !!注意该值仅仅是一个设定的值，后面的新拓扑的节点数目有可能小于该值
        # v_remove_nodes_idx = []

        # # 方法(1)随机去点生成拓扑，并去掉度为0的节点
        # while self.v_DG.number_of_nodes() > self.v_num_nodes:
        #     r_idx = random.randint(0, self.num_nodes - 1)
        #     if r_idx in v_remove_nodes_idx:
        #         continue
        #     else:
        #         self.v_DG.remove_node(r_idx)
        #         v_remove_nodes_idx.append(r_idx)
        #         # idx_zero_degree = []
        #         for j in list(self.v_DG.nodes()):
        #             if self.v_DG.degree(j) == 0:
        #                 # idx_zero_degree.append(j)
        #                 self.v_DG.remove_node(j)
        #                 v_remove_nodes_idx.append(j)#[4, 6, 1, 9, 16, 2, 5, 13, 18, 8, 0]

        # # 方法(2)按照连接度由低到高排序，从连接度最低值的节点中随机去掉一个，生成拓扑
        # while self.v_DG.number_of_nodes() > self.v_num_nodes:
        #     # 找出连接度最小的节点list: min_value_keys
        #     sorted_keys = [key for key, value in sorted(dict(self.v_DG.degree).items(), key=lambda item: item[1])]
        #     min_value = dict(self.v_DG.degree)[sorted_keys[0]]
        #     sorted_items = sorted(dict(self.v_DG.degree).items(), key=lambda item: item[1])
        #     min_value_keys = [key for key, value in sorted_items if value == min_value]
        #     r_idx = random.choice(min_value_keys)
        #     if r_idx in v_remove_nodes_idx:
        #         continue
        #     else:
        #         self.v_DG.remove_node(r_idx)
        #         v_remove_nodes_idx.append(r_idx)
        #         # idx_zero_degree = []
        #         for j in list(self.v_DG.nodes()):
        #             if self.v_DG.degree(j) == 0:
        #                 # idx_zero_degree.append(j)
        #                 self.v_DG.remove_node(j)
        #                 v_remove_nodes_idx.append(j)

        # # 方法(3)按照连接度由低到高排序，从list[连接度前几低的节点+随机选取的几个节点]中随机选取一个节点去掉，生成拓扑
        # while self.v_DG.number_of_nodes() > self.v_num_nodes:
        #     # 找出连接度前几低的节点list: min_value_keys
        #     sorted_keys = [key for key, value in sorted(dict(self.v_DG.degree).items(), key=lambda item: item[1])]
        #     # min_value_keys = sorted_keys[:3]
        #     min_value_keys = sorted_keys[:2] + random.sample(sorted_keys, 1)
        #     r_idx = random.choice(min_value_keys)
        #     if r_idx in v_remove_nodes_idx:
        #         continue
        #     else:
        #         self.v_DG.remove_node(r_idx)
        #         v_remove_nodes_idx.append(r_idx)
        #         # idx_zero_degree = []
        #         for j in list(self.v_DG.nodes()):
        #             if self.v_DG.degree(j) == 0:
        #                 # idx_zero_degree.append(j)
        #                 self.v_DG.remove_node(j)
        #                 v_remove_nodes_idx.append(j)

        # print(sorted(v_remove_nodes_idx))
        # nx.draw_networkx(self.v_DG, arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

        # [2, 3, 4, 7, 8, 11, 14, 16, 17, 19, 22]
        # [0, 3, 4, 5, 6, 8, 11, 16, 17, 19, 22]
        # [1, 2, 3, 4, 8, 11, 14, 16, 17, 19, 22]
        # [0, 2, 3, 5, 6, 7, 11, 13, 14, 16, 18]
        # [0, 2, 3, 4, 5, 8, 11, 12, 14, 16, 21]
        # [0, 2, 3, 4, 5, 6, 8, 11, 13, 14, 18]
        # [1, 2, 3, 4, 5, 7, 8, 11, 16, 19, 22]
        # [0, 1, 2, 3, 4, 5, 8, 10, 11, 14, 16]
        # [0, 1, 4, 9, 10, 11, 16, 17, 18, 19, 22]
        # [0, 3, 4, 8, 11, 14, 16, 17, 18, 19, 22]
        # # self.v_DG.remove_nodes_from(v_remove_nodes_idx)

        # self.v_num_nodes = self.v_DG.number_of_nodes()
        # logging.info("v_num_nodes = " + str(self.v_num_nodes))

        attach_word = "_" + str_v_net_name + "_node_num_" + str(self.v_num_nodes) \
                      + "_re_" + "_".join(str(x) for x in v_remove_nodes_idx)

        if config.topology_file == 'Abilene':  # *#
            self.topology = AbileneTopology(config, self.data_dir)
            self.traffic = AbileneTraffic(config, self.topology.num_nodes, self.data_dir)
        elif config.topology_file == 'GEANT':  # *#
            self.topology = utlis.network_env.ReGEANTTopology(v_remove_nodes_idx, config, self.data_dir, attach_word)
            self.traffic = utlis.network_env.ReGEANTTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Xspedius':  # *#
            self.topology = utlis.network_env.XspediusTopology(config, self.data_dir)
            self.traffic = utlis.network_env.XspediusTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'SwitchL3':  # *#
            self.topology = utlis.network_env.SwitchL3Topology(config, self.data_dir)
            self.traffic = utlis.network_env.SwitchL3Traffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Uunet':  # *#
            self.topology = utlis.network_env.UunetTopology(config, self.data_dir)
            self.traffic = utlis.network_env.UunetTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Dfn':
            self.topology = utlis.network_env.DfnTopology(config, self.data_dir)
            self.traffic = utlis.network_env.DfnTraffic(config, self.topology, self.data_dir)
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
        ##
        self.neighbors_idx = {i: list(self.DG.neighbors(i)) for i in range(self.num_nodes)}  # 邻居节点
        self.neighbors_idx_2_order = {}  # 二阶邻居节点
        for key, value in self.neighbors_idx.items():
            neighbors_2o = []
            for i_2o in value:
                neighbors_2o = neighbors_2o + self.neighbors_idx[i_2o]
            nodes_exc = [key] + value
            neighbors_2o = [x for x in neighbors_2o if x not in nodes_exc]  # 剔除源节点和一阶邻居节点
            # 剔除重复出现的数字，并从小到大排序
            unique_elements = []
            seen = set()
            for x in neighbors_2o:
                if x not in seen:
                    unique_elements.append(x)
                    seen.add(x)
            self.neighbors_idx_2_order[key] = sorted(unique_elements)

        self.node_id_to_idx = self.topology.node_id_to_idx
        self.node_id_list = self.topology.node_id_list
        self.link_id_list = self.topology.link_id_list  ## 注意links乱序，不建议使用
        self.link_idx_list = self.topology.link_idx_list  ## 注意links乱序，不建议使用

        self._get_ecmp_next_hops()  # self.ecmp_next_hops

        self.flow_pairs = [p for p in range(self.num_pairs)]

        self.load_multiplier = {}

        ##  删除 拓扑文件、最短路径文件、流量矩阵（TM）文件
        file_path = self.topology.topology_file
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = self.topology.shortest_paths_file
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = self.traffic.traffic_file
        if os.path.exists(file_path):
            os.remove(file_path)

        # if os.path.exists(file_path):
        #     # 使用os.remove来删除文件
        #     try:
        #         os.remove(file_path)
        #         print(f"文件 {file_path} 已被删除。")
        #     except OSError as e:
        #         # 如果删除过程中发生错误（比如没有权限），则打印错误信息
        #         print(f"删除文件时发生错误: {e.strerror}")
        #     else:
        #         print(f"文件 {file_path} 不存在。")

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


class InitVnetwork(object):

    def __init__(self, config, data_dir, v_DG, str_v_net_name):

        self.data_dir = data_dir
        self.v_DG = v_DG

        self.v_num_nodes = 12  # !!注意该值仅仅是一个设定的值，后面的新拓扑的节点数目有可能小于该值
        v_remove_nodes_idx = []

        # # 方法(1)随机去点生成拓扑，并去掉度为0的节点
        # while self.v_DG.number_of_nodes() > self.v_num_nodes:
        #     r_idx = random.randint(0, self.num_nodes - 1)
        #     if r_idx in v_remove_nodes_idx:
        #         continue
        #     else:
        #         self.v_DG.remove_node(r_idx)
        #         v_remove_nodes_idx.append(r_idx)
        #         # idx_zero_degree = []
        #         for j in list(self.v_DG.nodes()):
        #             if self.v_DG.degree(j) == 0:
        #                 # idx_zero_degree.append(j)
        #                 self.v_DG.remove_node(j)
        #                 v_remove_nodes_idx.append(j)#[4, 6, 1, 9, 16, 2, 5, 13, 18, 8, 0]

        # # 方法(2)按照连接度由低到高排序，从连接度最低值的节点中随机去掉一个，生成拓扑
        # while self.v_DG.number_of_nodes() > self.v_num_nodes:
        #     # 找出连接度最小的节点list: min_value_keys
        #     sorted_keys = [key for key, value in sorted(dict(self.v_DG.degree).items(), key=lambda item: item[1])]
        #     min_value = dict(self.v_DG.degree)[sorted_keys[0]]
        #     sorted_items = sorted(dict(self.v_DG.degree).items(), key=lambda item: item[1])
        #     min_value_keys = [key for key, value in sorted_items if value == min_value]
        #     r_idx = random.choice(min_value_keys)
        #     if r_idx in v_remove_nodes_idx:
        #         continue
        #     else:
        #         self.v_DG.remove_node(r_idx)
        #         v_remove_nodes_idx.append(r_idx)
        #         # idx_zero_degree = []
        #         for j in list(self.v_DG.nodes()):
        #             if self.v_DG.degree(j) == 0:
        #                 # idx_zero_degree.append(j)
        #                 self.v_DG.remove_node(j)
        #                 v_remove_nodes_idx.append(j)

        # 方法(3)按照连接度由低到高排序，从list[连接度前几低的节点+随机选取的几个节点]中随机选取一个节点去掉，生成拓扑
        while self.v_DG.number_of_nodes() > self.v_num_nodes:
            # 找出连接度前几低的节点list: min_value_keys
            sorted_keys = [key for key, value in sorted(dict(self.v_DG.degree).items(), key=lambda item: item[1])]
            # min_value_keys = sorted_keys[:3]
            min_value_keys = sorted_keys[:2] + random.sample(sorted_keys, 1)
            r_idx = random.choice(min_value_keys)
            if r_idx in v_remove_nodes_idx:
                continue
            else:
                self.v_DG.remove_node(r_idx)
                v_remove_nodes_idx.append(r_idx)
                # idx_zero_degree = []
                for j in list(self.v_DG.nodes()):
                    if self.v_DG.degree(j) == 0:
                        # idx_zero_degree.append(j)
                        self.v_DG.remove_node(j)
                        v_remove_nodes_idx.append(j)

        # print(sorted(v_remove_nodes_idx))
        # nx.draw_networkx(self.v_DG, arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

        # [2, 3, 4, 7, 8, 11, 14, 16, 17, 19, 22]
        # [0, 3, 4, 5, 6, 8, 11, 16, 17, 19, 22]
        # [1, 2, 3, 4, 8, 11, 14, 16, 17, 19, 22]
        # [0, 2, 3, 5, 6, 7, 11, 13, 14, 16, 18]
        # [0, 2, 3, 4, 5, 8, 11, 12, 14, 16, 21]
        # [0, 2, 3, 4, 5, 6, 8, 11, 13, 14, 18]
        # [1, 2, 3, 4, 5, 7, 8, 11, 16, 19, 22]
        # [0, 1, 2, 3, 4, 5, 8, 10, 11, 14, 16]
        # [0, 1, 4, 9, 10, 11, 16, 17, 18, 19, 22]
        # [0, 3, 4, 8, 11, 14, 16, 17, 18, 19, 22]
        # # self.v_DG.remove_nodes_from(v_remove_nodes_idx)

        self.v_num_nodes = self.v_DG.number_of_nodes()
        logging.info("v_num_nodes = " + str(self.v_num_nodes))

        attach_word = "_" + str_v_net_name + "_node_num_" + str(self.v_num_nodes) \
                      + "_re_" + "_".join(str(x) for x in v_remove_nodes_idx)

        if config.topology_file == 'Abilene':  # *#
            self.topology = AbileneTopology(config, self.data_dir)
            self.traffic = AbileneTraffic(config, self.topology.num_nodes, self.data_dir)
        elif config.topology_file == 'GEANT':  # *#
            self.topology = utlis.network_env.ReGEANTTopology(v_remove_nodes_idx, config, self.data_dir, attach_word)
            self.traffic = utlis.network_env.ReGEANTTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Xspedius':  # *#
            self.topology = utlis.network_env.XspediusTopology(config, self.data_dir)
            self.traffic = utlis.network_env.XspediusTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'SwitchL3':  # *#
            self.topology = utlis.network_env.SwitchL3Topology(config, self.data_dir)
            self.traffic = utlis.network_env.SwitchL3Traffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Uunet':  # *#
            self.topology = utlis.network_env.UunetTopology(config, self.data_dir)
            self.traffic = utlis.network_env.UunetTraffic(config, self.topology, self.data_dir)
        elif config.topology_file == 'Dfn':
            self.topology = utlis.network_env.DfnTopology(config, self.data_dir)
            self.traffic = utlis.network_env.DfnTraffic(config, self.topology, self.data_dir)
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
        ##
        self.neighbors_idx = {i: list(self.DG.neighbors(i)) for i in range(self.num_nodes)}  # 邻居节点
        self.neighbors_idx_2_order = {}  # 二阶邻居节点
        for key, value in self.neighbors_idx.items():
            neighbors_2o = []
            for i_2o in value:
                neighbors_2o = neighbors_2o + self.neighbors_idx[i_2o]
            nodes_exc = [key] + value
            neighbors_2o = [x for x in neighbors_2o if x not in nodes_exc]  # 剔除源节点和一阶邻居节点
            # 剔除重复出现的数字，并从小到大排序
            unique_elements = []
            seen = set()
            for x in neighbors_2o:
                if x not in seen:
                    unique_elements.append(x)
                    seen.add(x)
            self.neighbors_idx_2_order[key] = sorted(unique_elements)

        self.node_id_to_idx = self.topology.node_id_to_idx
        self.node_id_list = self.topology.node_id_list
        self.link_id_list = self.topology.link_id_list  ## 注意links乱序，不建议使用
        self.link_idx_list = self.topology.link_idx_list  ## 注意links乱序，不建议使用

        self._get_ecmp_next_hops()  # self.ecmp_next_hops

        self.flow_pairs = [p for p in range(self.num_pairs)]

        self.load_multiplier = {}

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


class EvalCustomMultiAgentEnv(CustomMultiAgentEnv):

    def __init__(self, config):
        super().__init__(config)


class CustomMAPPO_Agents(MAPPO_Agents):
    """The custom implementation of MAPPO agents.
    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
        custom_param: an additional custom parameter.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv,
                 # path_reward_save: str
                 ):
        # 调用父类的构造函数
        super(CustomMAPPO_Agents, self).__init__(config, envs)
        # 初始化自定义参数
        # self.custom_param = custom_param
        ##*#
        self.eval_max_step = config.eval_max_step
        # self.path_reward_save = path_reward_save
        self.config = config
        ##*#

    def test(self, env_fn, n_episodes):
        """
        Test the model for some episodes.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes to test.

        Returns:
        """
        scores = self.test_run_episodes(env_fn=env_fn, n_episodes=n_episodes, test_mode=True)
        return scores

    def test_run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        """
        Run some episodes when use RNN.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes.
            test_mode (bool): Whether to test the model.

        Returns:
            Scores: The episode scores.
        """
        eval_step_count = 0

        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_count, scores, best_score = 0, [0.0 for _ in range(num_envs)], -np.inf
        obs_dict, info = envs.reset()
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        state = envs.buf_state if self.use_global_state else None
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_hidden_actor, rnn_hidden_critic = self.init_rnn_hidden(num_envs)

        #
        ar_mean_list = []
        V_mean_list = []
        V_k1_mean_list = []
        V_g_mean_list = []
        while episode_count < n_episodes:
            step_info = {}
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions,
                                     rnn_hidden_actor=rnn_hidden_actor, rnn_hidden_critic=rnn_hidden_critic,
                                     test_mode=test_mode)
            rnn_hidden_actor, rnn_hidden_critic = policy_out['rnn_hidden_actor'], policy_out['rnn_hidden_critic']
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                      terminated_dict, info, **{'state': state})
            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            state = envs.buf_state if self.use_global_state else None

            for i in range(num_envs):
                eval_step_count += 1
                if eval_step_count % self.eval_max_step == 0:
                    episode_count += 1
                    with open(self.config.path_reward_save + '/' + self.config.save_attach_word + 'eval_marl_env.pkl',
                              'rb') as handle:
                        reward_k1_list, reward_k2_list, reward_k3_list, reward_g_list, g_done_list, dict_eval_index \
                            = pickle.load(handle)

                    ar = np.array(g_done_list[-512 * 1:], dtype=bool).astype(int)
                    print('Acceptance_rate_mean: ' + str(ar.mean()) + '; '
                          + 'Acceptance_rate_std: ' + str(ar.std()) + '; ')
                    ar_mean_list.append(ar.mean())
                    V_mean_list.append(np.array(dict_eval_index['V'][-512 * 1:]).mean())
                    V_k1_mean_list.append(np.array(dict_eval_index['V_k1'][-512 * 1:]).mean())
                    V_g_mean_list.append(np.array(dict_eval_index['V_g'][-512 * 1:]).mean())

                    os.remove(self.config.path_reward_save + '/' + self.config.save_attach_word + 'eval_marl_env.pkl')
                else:
                    pass

        # with open(self.config.path_reward_save + '/' + self.config.save_attach_word + 'dict_train_index_marl_env.pkl',
        #           'rb') as handle:
        #     dict_train_index = pickle.load(handle)

        # L = 512  # 512*4
        # r = np.array(dict_train_index["V"])
        # r_mean_L = []
        # for i in range(int(np.floor(len(r) / L))):
        #     r_mean_L.append(r[i * L:i * L + L].mean())
        # r_mean_L[0] = r_mean_L[2];r_mean_L[1] = r_mean_L[2]
        # r_mean_L[1] = r_mean_L[2]
        # plt.plot(r_mean_L[:])
        # plt.title('reward_k1_list r_mean_L')
        # plt.show()

        # bool_array = np.array(g_done_list[:512000], dtype=bool)
        # int_array = bool_array.astype(int)
        # g_r = np.array(int_array)
        # rr = []
        # L_mean = 512  # 512
        # for i in range(int(np.ceil(len(g_done_list) / L_mean))):
        #     arr = g_r[i * L_mean:i * L_mean + L_mean]
        #     rr.append((len(arr) - np.sum(arr == 0.0)) / len(arr))
        # rr[0] = rr[2]
        # rr[1] = rr[2]
        # plt.title('rr')
        # plt.plot(rr[:])
        # plt.ylim(0, 1)
        # plt.show()

        print('.........')
        print('ar_mean_list-mean: ' + str(np.array(ar_mean_list).mean()) + '; '
              + 'ar_mean_list-std: ' + str(np.array(ar_mean_list).std()) + '; ')
        print('V_mean_list-mean: ' + str(np.array(V_mean_list).mean()) + '; '
              + 'V_mean_list-std: ' + str(np.array(V_mean_list).std()) + '; ')
        print('V_k1_mean_list-mean: ' + str(np.array(V_k1_mean_list).mean()) + '; '
              + 'V_k1_mean_list-std: ' + str(np.array(V_k1_mean_list).std()) + '; ')
        print('V_g_mean_list-mean: ' + str(np.array(V_g_mean_list).mean()) + '; '
              + 'V_g_mean_list-std: ' + str(np.array(V_g_mean_list).std()) + '; ')
        print("END")

        # if eval_step_count == self.config.eval_max_episode:
                #     pass

                # if all(terminated_dict[i].values()) or truncated[i]:
                #     episode_count += 1
                #     episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                #     scores.append(episode_score)
                #     if test_mode:
                #         if self.use_rnn:
                #             rnn_hidden_actor, _ = self.init_hidden_item(i, rnn_hidden_actor)
                #         if best_score < episode_score:
                #             best_score = episode_score
                #             episode_videos = videos[i].copy()
                #         if self.config.test_mode:
                #             print("Episode: %d, Score: %.2f" % (episode_count, episode_score))
                #     else:
                #         if all(terminated_dict[i].values()):
                #             value_next = {key: 0.0 for key in self.agent_keys}
                #         else:
                #             _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i], state=state[i],
                #                                              rnn_hidden_critic=rnn_hidden_critic)
                #         self.memory.finish_path(i_env=i, i_step=info[i]['episode_step'], value_next=value_next,
                #                                 value_normalizer=self.learner.value_normalizer)
                #         if self.use_rnn:
                #             rnn_hidden_actor, rnn_hidden_critic = self.init_hidden_item(i, rnn_hidden_actor,
                #                                                                         rnn_hidden_critic)
                #         if self.use_wandb:
                #             step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                #             step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                #         else:
                #             step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                #             step_info["Train-Results/Episode-Rewards"] = {
                #                 "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                #         self.current_step += info[i]["episode_step"]
                #         self.log_infos(step_info, self.current_step)
                #     obs_dict[i] = info[i]["reset_obs"]
                #     envs.buf_obs[i] = info[i]["reset_obs"]
                #     if self.use_actions_mask:
                #         avail_actions[i] = info[i]["reset_avail_actions"]
                #         envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]

        # if test_mode:
        #     if self.config.render_mode == "rgb_array" and self.render:
        #         # time, height, width, channel -> time, channel, height, width
        #         videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
        #         self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)
        #
        #     if self.config.test_mode:
        #         print("Best Score: %.2f" % best_score)
        #
        #     test_info = {
        #         "Test-Results/Episode-Rewards/Mean-Score": np.mean(scores),
        #         "Test-Results/Episode-Rewards/Std-Score": np.std(scores),
        #     }
        #     self.log_infos(test_info, self.current_step)
        #     if env_fn is not None:
        #         envs.close()

        scores = None
        return scores


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: IPPO for MPE.")
    parser.add_argument("--env-id", type=str, default="new_env_id")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="configs_main.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)

    topology_file = configs_dict['topology_file']
    traffic_file = 'None'
    test_traffic_file = 'None'
    train_traffic_dir = 'None'
    test_traffic_dir = 'None'

    if topology_file == 'Abilene':
        traffic_file = 'Abilene_TM1'
        test_traffic_file = 'Abilene_TM2'
    elif topology_file == 'GEANT':
        traffic_file = 'GEANT_TM1'
        test_traffic_file = 'GEANT_TM2'
        train_traffic_dir = '2005_04_1st_week'
        test_traffic_dir = '2005_04_2nd_week'

    configs_dict['traffic_file'] = traffic_file
    configs_dict['test_traffic_file'] = test_traffic_file
    configs_dict['train_traffic_dir'] = train_traffic_dir
    configs_dict['test_traffic_dir'] = test_traffic_dir

    # path set
    result_root = configs_dict['root'] + '/result'
    data_root = configs_dict['root'] + '/data'
    configs_dict['result_root'] = result_root
    configs_dict['data_root'] = data_root

    path_model_train_log = os.path.join(result_root, 'model_train_log')
    if not os.path.exists("%s" % path_model_train_log):
        os.makedirs("%s" % path_model_train_log)
    configs_dict['path_model_train_log'] = path_model_train_log

    path_model_save = os.path.join(result_root, 'model_save')
    if not os.path.exists("%s" % path_model_save):
        os.makedirs("%s" % path_model_save)
    configs_dict['path_model_save'] = path_model_save

    path_env_save = os.path.join(result_root, 'env_save')
    if not os.path.exists("%s" % path_env_save):
        os.makedirs("%s" % path_env_save)
    configs_dict['path_env_save'] = path_env_save

    path_reward_save = os.path.join(result_root, 'reward_save')
    if not os.path.exists("%s" % path_reward_save):
        os.makedirs("%s" % path_reward_save)
    configs_dict['path_reward_save'] = path_reward_save

    # path_Agents_save = os.path.join(result_root, 'Agents_save')
    # if not os.path.exists("%s" % path_Agents_save):
    #     os.makedirs("%s" % path_Agents_save)
    # configs_dict['path_Agents_save'] = path_Agents_save

    path_F_save = os.path.join(result_root, 'F_save')
    if not os.path.exists("%s" % path_F_save):
        os.makedirs("%s" % path_F_save)
    configs_dict['path_F_save'] = path_F_save

    path_logger = os.path.join(result_root, 'logger_save')
    if not os.path.exists("%s" % path_logger):
        os.makedirs("%s" % path_logger)
    path_logger = os.path.join(result_root, 'logger_save')

    # ----------------------------logging------------------------
    setlogger(os.path.join(path_logger, 'train.log'))
    f = open("%s/train.log" % path_logger, 'w')
    f.truncate()
    f.close()

    logging.info("Create the system save path of model and env")

    save_attach_word = str(configs_dict['max_num_steps']) + '_' \
                       + configs_dict['topology_file'] + '_pr-900-30' + '_' # ！！
    configs_dict['save_attach_word'] = save_attach_word

    configs = argparse.Namespace(**configs_dict)

    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = CustomMultiAgentEnv
    set_seed(configs.seed)  # Set the random seed.
    # envs = make_envs(configs)  # Make the environment.
    #
    # Agents = CustomMAPPO_Agents(config=configs, envs=envs, path_reward_save=path_reward_save)  # Create the Independent PPO agents.

    # train_information = {"Deep learning toolbox": configs.dl_toolbox,
    #                      "Calculating device": configs.device,
    #                      "Algorithm": configs.agent,
    #                      "Environment": configs.env_name,
    #                      "Scenario": configs.env_id}
    # for k, v in train_information.items():  # Print the training information.
    #     print(f"{k}: {v}")

    if configs.benchmark:
        # def env_fn():  # Define an environment function for test method.
        #     configs_test = deepcopy(configs)
        #     configs_test.parallels = configs_test.test_episode
        #     configs_test.running_steps = 1024
        #     configs_test.eval_interval = 1024
        #     envs_eval = make_envs(configs_test)
        #     envs_eval.max_episode_steps = 100
        #     return envs_eval
            # return make_envs(configs_test)

        # train_steps = configs.running_steps // configs.parallels
        # eval_interval = configs.eval_interval // configs.parallels
        # test_episode = configs.test_episode
        # num_epoch = int(train_steps / eval_interval)

        marl_model_dir = 'dir_model_pr-900-30' # ！！ 主要更改它 # 为避免load错误模型，dir 存放一个.pth的model文件
        # marl_model_dir = 'dir_model_1'  # 主要更改它 # 为避免load错误模型，dir 存放一个.pth的model文件
        marl_model_name = save_attach_word + "mappo_model_save" + ".pth"
        model_dir_save = os.path.join(os.getcwd(), configs.model_dir, marl_model_dir)  # 更改model保存路径文件夹
        # Agents.model_dir_load = configs.model_dir  # 更改model加载路径文件夹

        path_marl_model = os.path.join(model_dir_save, marl_model_name)
        path_marl_env = path_env_save + '/' + save_attach_word + 'env_save.pkl'
        if os.path.exists(path_marl_model) == False or os.path.exists(path_marl_env) == False:
            envs = make_envs(configs)  # Make the environment.
            Agents = CustomMAPPO_Agents(config=configs, envs=envs)  # Create the Independent PPO agents.

            Agents.model_dir_save = model_dir_save

            Agents.train(configs.max_num_steps)
            # 保存 model env
            Agents.save_model(model_name=marl_model_name)
            with open(path_marl_env, 'wb') as handle:
                pickle.dump((Agents.envs), handle)
        else:
            # 加载 model env
            with open(path_marl_env, 'rb') as handle:
                envs_save = pickle.load(handle)
            Agents = CustomMAPPO_Agents(config=configs, envs=envs_save)  # Create the Independent PPO agents.

            Agents.model_dir_save = model_dir_save
            Agents.load_model(path=Agents.model_dir_load, model=marl_model_dir)

        test_scores = Agents.test(env_fn=None, n_episodes=configs.eval_max_episode)
        # test_scores = Agents.test(env_fn=envs_save, n_episodes=configs.eval_max_episode)


        # best_scores_info = {"mean": np.mean(test_scores),
        #                     "std": np.std(test_scores),
        #                     }
        # for i_epoch in range(num_epoch):
        #     print("Epoch: %d/%d:" % (i_epoch, num_epoch))
        #     Agents.train(eval_interval)
        #     test_scores = Agents.test(env_fn, test_episode)
        #
        #     if np.mean(test_scores) > best_scores_info["mean"]:
        #         best_scores_info = {"mean": np.mean(test_scores),
        #                             "std": np.std(test_scores),
        #                             }
        #         # save best model
        #         Agents.save_model(model_name="best_model.pth")
        # # end benchmarking
        # print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        pass
        # if configs.test:
        #     def env_fn():
        #         configs.parallels = configs.test_episode
        #         return make_envs(configs)
        #
        #
        #     Agents.load_model(path=Agents.model_dir_load)
        #     scores = Agents.test(env_fn, configs.test_episode)
        #     print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
        #     print("Finish testing.")
        # else:
        #     Agents.train(configs.running_steps // configs.parallels)
        #     Agents.save_model("final_train_model.pth")
        #     print("Finish training!")

    Agents.finish()
