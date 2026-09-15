#! /usr/bin/env python

import os
import pickle
import logging
import heapq
import math

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from utlis.general_function import setlogger, setup_seed
from utlis.config import get_config
from utlis.models import CustomActorCriticPolicy
from utlis.custom_environment_drlte import CustomEnv, eval_env_step#, env_init_eval#, env_init_real_TM_eval, env_init_simu_TM_eval,

'''展开 # 展开当前代码块ctrl =  # 彻底展开当前代码块ctrl alt =  # 展开所有代码块ctrl shift +
折叠 # 折叠当前代码块ctrl -  # 彻底折叠当前代码块ctrl alt -  # 折叠所有代码块ctrl shift - '''


def model_train(env_save_pkl, model_save_pkl, link_num_ratio, flow_num_ratio):

    logging.info("Begin medel_train")
    setup_seed(seed=1)

    env = CustomEnv(path_reward_save=path_reward_save,
                    link_num_ratio=link_num_ratio,# 0.15#更改关键links和关键flow数量参数
                    flow_num_ratio=flow_num_ratio,# 0.15#
                    is_training=True)
    env.total_steps = 512*500 #504*1500  # 1500 episodes => 768000## 2000 episodes => 1024000

    model = PPO(
        policy=CustomActorCriticPolicy, #policy='MlpPolicy', #CustomActorCriticPolicy,  #MlpPolicy, 定义策略网络为MLP网络                       # MlpPolicy定义策略网络为MLP网络
        env=env,
        learning_rate=3e-4,
        n_steps=512,#168=672/4 #504 #512
        batch_size=64,#63 #64
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        # policy_kwargs={"net_arch": net_arch},
        verbose=1,                                   # verbose=1代表打印训练信息，如果是0为不打印，2为打印调试信息
        tensorboard_log=path_model_train_log,  # 训练数据保存目录，可以用tensorboard查看
        seed=None
    )
    model.learn(total_timesteps=env.total_steps)

    model.save(model_save_pkl)
    # del model  # remove to demonstrate saving and loading
    with open(env_save_pkl, 'wb') as handle:
        pickle.dump((env), handle)


def model_eval(env, model):

    # setup_seed(SEED)

    obs = env.state
    env.time_list = []
    env.traffic_r_l = []
    env.flow_r_l = []

    if env.tm_idx != 0:#空跑一定次数的step，使得 tm_idx为 0，注意先调整到正确的num_crit_links，K_num，法等
        num_dry_run = env.tm_cnt
        for ndr in range(num_dry_run):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if env.tm_idx == 0:
                break

    assert env.tm_idx == 0, 'ERROR, because tm_idx is not 0'

    # dict_num_try_result = dict()
    num_try = 1#尝试10个episode，（重复实验次数）
    for k_episode in range(num_try):

        # MLU
        env.norm_mlu_eval_list = []
        env.norm_ecmp_mlu_eval_list = []
        # time
        env.time_list = []
        env.traffic_r_l = []
        env.flow_r_l = []

        num_step_eval = env.tm_cnt
        print('num_step_eval = %d' % num_step_eval)
        for j in range(num_step_eval):
            action, _states = model.predict(obs)
            # action = np.array([1, 0.8, 0.6, 0.4, 0.2, 0]).astype(np.float32)
            # action = np.random.rand(6).astype(np.float32)
            obs, rewards, dones, info = eval_env_step(env, action)
            print(str(j))

        norm_mlu_eval = np.array(env.norm_mlu_eval_list)
        norm_ecmp_mlu_eval = np.array(env.norm_ecmp_mlu_eval_list)

        print('average_norm_mlu_eval '+str(norm_mlu_eval.mean())+'; '
              +'average_norm_ecmp_mlu_eval'+str(norm_ecmp_mlu_eval.mean())+'; ')

        traffic_r_l = np.array(env.traffic_r_l)
        flow_r_l = np.array(env.flow_r_l)
        time_array= np.array(env.time_list)

        import pandas as pd
        path_plt_data = os.path.join(root, 'result/plt_data')
        if not os.path.exists("%s" % path_plt_data):
            os.makedirs("%s" % path_plt_data)
        # 字典中的key值即为csv中列名
        L = int(len(traffic_r_l) / 7)
        df = pd.DataFrame( {'traffic_r_l': traffic_r_l[L * 3:L * 4], 'flow_r_l': flow_r_l[L * 3:L * 4]})#,
        # 将DataFrame存储为csv,index表示是否显示行名，default=True;
        # 在参数header和index中使用True或False指定是否指定header（列名，pandas.DataFrame的列）和index（行名，pandas.DataFrame的索引）。默认值为True。
        df.to_csv(path_plt_data + '/' + 'SwitchL3_tr_fl_r_DRL-TE.csv', header=True, index=False,
                  float_format='%.5f')  # index为true会多写一列index,分隔符默认为逗号 #, sep='\t',

    print('END')

    # L = int(env.tm_cnt/7)
    # rn = norm_mlu_eval
    # e = norm_ecmp_mlu_eval
    # rn_mean_L = []
    # e_mean_L = []
    # for i in range(int(np.floor(len(rn) / L))):
    #     rn_mean_L.append(rn[i * L:i * L + L].mean())
    # for i in range(int(np.floor(len(e) / L))):
    #     e_mean_L.append(e[i * L:i * L + L].mean())
    # plt.plot(rn_mean_L)
    # plt.title('GENAT')
    # plt.plot(e_mean_L)
    # plt.ylim(0, 1)
    # plt.show()

    # L = 512
    # a = np.array(env.r_mlu_list);
    # b = np.array(env.optimal_mlu_list);
    # c = b / a;
    # reward = (c - 0.4) * env.reward_scaling
    # reward_mean_L = []
    # for i in range(int(np.floor(len(reward) / L))):
    #     reward_mean_L.append(reward[i * L:i * L + L].mean())
    # plt.plot(reward_mean_L)
    # plt.title('reward_mean_L.pkl')
    # plt.ylim(0, 1)
    # plt.show()
    # c_mean_L = []
    # for i in range(int(np.floor(len(c) / L))):
    #     c_mean_L.append(c[i * L:i * L + L].mean())
    # plt.plot(c_mean_L)
    # plt.title('norm_mlu_mean_L.pkl')
    # plt.ylim(0, 1)
    # plt.show()


        # dict_num_try_result[k_episode]={
        #                                 'average_comp_queu_time': average_comp_queu_time,
        #                                 'average_cache_access_delay': average_cache_access_delay}

    # list_dr = []
    # for i in range(len(dict_num_try_result)):
    #     dr = dict_num_try_result[i]
    #     print('k_episode='+str(i)+' --> '+ 'average_task_completion_time= '  +str(dr['average_task_completion_time']))
    #     list_dr.append(dr['average_task_completion_time'])
    # print('mean = '+str(np.array(list_dr).mean())+'; '+'var = '+str(np.array(list_dr).var()))

def main():

    # attach_word_list = ['Random', 'Node-based', 'Centrality-based']
    attach_word_list = ['DRL-TE']
    for i_l in attach_word_list:
        attach_word = '_' + i_l + '_'
        model_save_pkl = os.path.join(path_model_save, config.topology_file + attach_word + 'model_save.pkl')
        env_save_pkl = os.path.join(path_env_save_save, config.topology_file + attach_word + 'env_save.pkl')

        if not os.path.exists("%s" % model_save_pkl) or not os.path.exists("%s" % env_save_pkl):
            logging.info("———————生成env和DRL_model—————")
            model_train(env_save_pkl, model_save_pkl, link_num_ratio=0.15, flow_num_ratio=0.15)
        else:
            logging.info("The data folders of the datapath already exists. Load directly.")
            model = PPO.load(model_save_pkl)
            with open(env_save_pkl, 'rb') as pkl:
                env = pickle.load(pkl)
        model_eval(env, model)


if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------------path---------------------------
    config = get_config()
    root = config.root
    result_root = config.result_root
    data_root = config.data_root

    path_model_train_log = os.path.join(result_root, 'model_train_log')
    if not os.path.exists("%s" % path_model_train_log):
        os.makedirs("%s" % path_model_train_log)

    path_model_save = os.path.join(result_root, 'model_save')
    if not os.path.exists("%s" % path_model_save):
        os.makedirs("%s" % path_model_save)

    path_env_save_save = os.path.join(result_root, 'env_save')
    if not os.path.exists("%s" % path_env_save_save):
        os.makedirs("%s" % path_env_save_save)

    path_reward_save = os.path.join(result_root, 'reward_save')
    if not os.path.exists("%s" % path_reward_save):
        os.makedirs("%s" % path_reward_save)

    path_F_save = os.path.join(result_root, 'F_save')
    if not os.path.exists("%s" % path_F_save):
        os.makedirs("%s" % path_F_save)

    path_logger = os.path.join(result_root, 'logger_save')
    if not os.path.exists("%s" % path_logger):
        os.makedirs("%s" % path_logger)

    # ----------------------------logging------------------------
    setlogger(os.path.join(path_logger, 'train.log'))
    f = open("%s/train.log" % path_logger, 'w')
    f.truncate()
    f.close()

    logging.info("Create the system save path of model and env")

    main()
