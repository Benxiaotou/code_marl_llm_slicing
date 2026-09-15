import torch

class NetworkConfig(object):
    scale = 100

    max_step = 1000 * scale
  
    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1

    save_step = 10 * scale
    max_to_keep = 1000

    Conv2D_out = 128
    Dense_out = 128
  
    optimizer = 'RMSprop'
    #optimizer = 'Adam'
    
    logit_clipping = 10       #10 or 0, = 0 means logit clipping is disabled


class Config(NetworkConfig):
    version = 'TE_v2'

    project_name = 'CFR-RL'

    method = 'actor_critic'
    #method = 'pure_policy'
  
    model_type = 'Conv'

    topology_file = 'GEANT' #'Xspedius'#'Abilene'#'GEANT'#Uunet#SwitchL3

    if topology_file == 'Abilene':
        traffic_file = 'Abilene_TM1'
        test_traffic_file = 'Abilene_TM2'
    elif topology_file == 'GEANT':
        traffic_file = 'GEANT_TM1'
        test_traffic_file = 'GEANT_TM2'
        train_traffic_dir = '2005_04_1st_week'
        test_traffic_dir = '2005_04_2nd_week'
    elif topology_file == 'Xspedius':
        traffic_file = 'Xspedius_TM1'
        test_traffic_file = 'Xspedius_TM2'
    elif topology_file == 'SwitchL3':
        traffic_file = 'SwitchL3_TM1'
        test_traffic_file = 'SwitchL3_TM2'
    elif topology_file == 'Uunet':
        traffic_file = 'Uunet_TM1'
        test_traffic_file = 'Uunet_TM2'

    elif topology_file == 'BtNorthAmerica':
        traffic_file = 'BtNorthAmerica_TM1'
        test_traffic_file = 'BtNorthAmerica_TM2'
    elif topology_file == 'Cernet':
        traffic_file = 'Cernet_TM1'
        test_traffic_file = 'Cernet_TM2'
    elif topology_file == 'Dfn':
        traffic_file = 'Dfn_TM1'
        test_traffic_file = 'Dfn_TM2'
    elif topology_file == 'ChinanetL3':
        traffic_file = 'Chinanet_TM1'
        test_traffic_file = 'Chinanet_TM2'
    else:
        pass


    tm_history = 1

    max_moves = 10            #percentage

   # For pure policy
    baseline = 'avg'          #avg, best

    num_steps_per_episode = int(2016/16) ##126,

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    root = "C:\\Users\\Lenovo\\Desktop\\论文_研究\\pxl_4\\code\\code_marl_v0"
    result_root = root + '/result'
    data_root = root + '/data'

def get_config(FLAGS=None):

    config = Config

    if not FLAGS == None:
        for k, v in FLAGS.__flags.items():
            if hasattr(config, k):
                setattr(config, k, v.value)

    return config
