from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from tqdm import tqdm
import numpy as np
import networkx as nx
import tmgen
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import heapq
import collections

MAX_PATH_NUM = 3


class AbileneTopology(object):

    def __init__(self, config, data_dir='./data/'):
        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'
        self.DG = nx.DiGraph()  ##https://www.osgeo.cn/networkx/reference/classes/digraph.html#overview

        self.load_topology()
        self.calculate_paths()

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = int(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), int(w))])

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.show()
        #
        # nx.draw(self.DG)

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.DG, s, d, weight='weight')))
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class AbileneTraffic(object):
    def __init__(self, config, num_nodes, data_dir='./data/', is_training=True):
        if is_training:
            self.traffic_file = data_dir + config.topology_file + '/' + config.traffic_file  #
        else:
            self.traffic_file = data_dir + config.topology_file + '/' + config.test_traffic_file  #
        self.num_nodes = num_nodes
        self.load_traffic(config)

    def load_traffic(self, config):
        assert os.path.exists(self.traffic_file)
        print('[*] Loading traffic matrices...', self.traffic_file)
        f = open(self.traffic_file, 'r')
        traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            assert total_volume_cnt == self.num_nodes * self.num_nodes
            matrix = np.zeros((self.num_nodes, self.num_nodes))
            for v in range(total_volume_cnt):
                i = int(v / self.num_nodes)
                j = v % self.num_nodes
                if i != j:
                    matrix[i][j] = float(volumes[v])
            # print(matrix + '\n')
            traffic_matrices.append(matrix)

        f.close()
        self.traffic_matrices = np.array(traffic_matrices)

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        print('AbileneTraffic matrices dims: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class GEANTTopology(object):

    def __init__(self, config, data_dir='./data/'):
        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'
        self.DG = nx.DiGraph()

        self.GEANT_data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        GEANT_topology_file_dir = self.GEANT_data_dir + 'topology_file'
        GEANT_topology_file_xml = GEANT_topology_file_dir + '/' + 'topology-anonymised.xml'

        # 解析 XML 文件
        tree = ET.parse(GEANT_topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for node in root[1][0].iter(tag='node'):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            node_id = node.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
            self.node_id_list.append(node_id)
            self.node_id_to_idx[node_id] = index
            index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        for link in root[1][1].iter(tag='link'):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            from_node_id = link[0].get(key='node')
            to_node_id = link[1].get(key='node')
            self.link_id_list.append((from_node_id, to_node_id))
            self.link_idx_list.append((self.node_id_to_idx[from_node_id], self.node_id_to_idx[to_node_id]))

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class GEANTTraffic(object):

    def __init__(self, config, topology, data_dir='./data/', is_training=True):
        if is_training:
            self.traffic_file = data_dir + config.topology_file + '/' + config.traffic_file  #
        else:
            self.traffic_file = data_dir + config.topology_file + '/' + config.test_traffic_file  #
        self.num_nodes = topology.num_nodes
        self.load_traffic(topology, config, is_training)

    def load_traffic(self, topology, config, is_training):
        if not os.path.exists(self.traffic_file):  #
            self.read_traffic(topology, config, is_training)

        print('[*] Loading traffic matrices...', self.traffic_file)
        f = open(self.traffic_file, 'r')
        traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            assert total_volume_cnt == self.num_nodes * self.num_nodes
            matrix = np.zeros((self.num_nodes, self.num_nodes))
            for v in range(total_volume_cnt):
                i = int(v / self.num_nodes)
                j = v % self.num_nodes
                if i != j:
                    matrix[i][j] = float(volumes[v])
            # print(matrix + '\n')
            traffic_matrices.append(matrix)

        f.close()
        self.traffic_matrices = np.array(traffic_matrices)

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        print('GEANTTraffic matrices dims: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))

    def read_traffic(self, topology, config, is_training):  ##
        '''
        取出traffic信息,并生成TM文件，文件中每一行为一个TM数据（529）
        '''
        if is_training:
            GEANT_traffic_file_dir = config.data_root + '/' + config.topology_file + '/' + config.train_traffic_dir
        else:
            GEANT_traffic_file_dir = config.data_root + '/' + config.topology_file + '/' + config.test_traffic_dir
        # 获取文件夹中的所有文件
        files = os.listdir(GEANT_traffic_file_dir)

        self.file_xml_count = len(files)
        assert self.file_xml_count == 672, "Note that the file_xml_count of '%s' is not 672" % GEANT_traffic_file_dir

        # 按文件名进行排序
        sorted_files = sorted(files)

        # 取出各个XML文件中的数据并按顺序组成多个TM，再写入一个共同的TM文件
        with open(self.traffic_file, 'w') as f:
            # 遍历排序后的文件列表
            for file_name in sorted_files:
                # 拼接文件的完整路径
                path_file_xml = os.path.join(GEANT_traffic_file_dir, file_name)
                # print(path_file_xml)

                read_TM = np.zeros((self.num_nodes, self.num_nodes))
                # 处理文件
                tree = ET.parse(path_file_xml)
                root = tree.getroot()
                for src in root[1].iter(tag='src'):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
                    src_id = src.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                    s = topology.node_id_to_idx[src_id]
                    for dst in src.iter(tag='dst'):
                        dst_id = dst.get(key='id')
                        d = topology.node_id_to_idx[dst_id]
                        demand_value = dst.text
                        read_TM[s][d] = demand_value

                # 将TM数据写入一个共同的TM文件，这里先将read_TM的shape(23, 23)转为write_TM的shape(529,)
                write_TM = read_TM.reshape(-1)
                f.write(' '.join([str(x) for x in write_TM]) + ' \n')


class BtNorthAmericaTopology(object):  # **#需更改类名称

    def __init__(self, config, data_dir='./data/'):
        self.DG = nx.DiGraph()

        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'

        self.data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        topology_file_dir = self.data_dir + 'topology_file'
        topology_file_xml = topology_file_dir + '/' + 'topology-zoo.org_files_BtNorthAmerica.xml'  # **# 需更改.xml文件名

        # 解析 XML 文件
        tree = ET.parse(topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='id') != None:
                # for g_tag in graph_tag.iter(tag=None):
                node_id = graph_tag.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                self.node_id_list.append(node_id)
                self.node_id_to_idx[node_id] = index
                index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        assert root[-1].get(key='edgedefault') == "undirected", "ERROE，请检查.xml文件的graph位置"
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='source') != None:
                source_node_id = graph_tag.get(key='source')
                target_node_id = graph_tag.get(key='target')
                if (source_node_id, target_node_id) in self.link_id_list:
                    continue
                self.link_id_list.append((source_node_id, target_node_id))
                self.link_id_list.append((target_node_id, source_node_id))  # 注意我们考虑的是单向link，所以zoo数据集中的每个link应该看成两条
                self.link_idx_list.append((self.node_id_to_idx[source_node_id], self.node_id_to_idx[target_node_id]))
                self.link_idx_list.append(
                    (self.node_id_to_idx[target_node_id], self.node_id_to_idx[source_node_id]))  # 同理

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class XspediusTopology(object):  # **#需更改类名称

    def __init__(self, config, data_dir='./data/'):
        self.DG = nx.DiGraph()

        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'

        self.data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        topology_file_dir = self.data_dir + 'topology_file'
        topology_file_xml = topology_file_dir + '/' + 'topology-zoo.org_files_Xspedius.xml'  # **# 需更改.xml文件名

        # 解析 XML 文件
        tree = ET.parse(topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='id') != None:
                # for g_tag in graph_tag.iter(tag=None):
                node_id = graph_tag.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                self.node_id_list.append(node_id)
                self.node_id_to_idx[node_id] = index
                index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        assert root[-1].get(key='edgedefault') == "undirected", "ERROE，请检查.xml文件的graph位置"
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='source') != None:
                source_node_id = graph_tag.get(key='source')
                target_node_id = graph_tag.get(key='target')
                if (source_node_id, target_node_id) in self.link_id_list:
                    continue
                self.link_id_list.append((source_node_id, target_node_id))
                self.link_id_list.append((target_node_id, source_node_id))  # 注意我们考虑的是单向link，所以zoo数据集中的每个link应该看成两条
                self.link_idx_list.append((self.node_id_to_idx[source_node_id], self.node_id_to_idx[target_node_id]))
                self.link_idx_list.append(
                    (self.node_id_to_idx[target_node_id], self.node_id_to_idx[source_node_id]))  # 同理

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class XspediusTraffic(object):  # **#需更改类名称

    def __init__(self, config, topology, data_dir='./data/', is_training=True):
        self.tm_save_file = data_dir + config.topology_file + '/' + config.topology_file + '_TM.pkl'
        if is_training:
            self.traffic_file = data_dir + config.topology_file + '/' + config.traffic_file  #
        else:
            self.traffic_file = data_dir + config.topology_file + '/' + config.test_traffic_file  #
        self.num_nodes = topology.num_nodes
        self.load_traffic(topology, config, is_training)

    def load_traffic(self, topology, config, is_training):

        print('[*] Generate traffic matrices...')

        if os.path.exists(self.tm_save_file):
            with open(self.tm_save_file, 'rb') as pkl:
                self.tm = pickle.load(pkl)
            # assert len(self.optimal_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
        else:
            # tm = tmgen.models.modulated_gravity_tm()
            print("TM模拟")
            mean_flow_traffic = 10000
            self.tm = modulated_gravity_tm(num_nodes=self.num_nodes,  #
                                           num_tms=672,  # 4*24*7 # 12*24*7=2016
                                           mean_traffic=mean_flow_traffic,  # TM中flow平均值的最小值
                                           pm_ratio=1.5,
                                           t_ratio=.75,
                                           diurnal_freq=6 / (24 * 24),  # 1 / 24,
                                           spatial_variance=mean_flow_traffic * 0.5,  # 单个TM内部变化
                                           temporal_variance=50)  # 100

            with open(self.tm_save_file, 'wb') as handle:
                pickle.dump(self.tm, handle)

        if is_training:
            self.traffic_matrices = self.tm#[:96*5]##训练
        else:
            self.traffic_matrices = self.tm[96*5:96*7]##测试

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]  # 注意它的设置
        print('XspediusTraffic matrices dims of training: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))

        # tms_shape = self.traffic_matrices_eval.shape
        # self.tm_cnt_eval = tms_shape[0]#注意它的设置
        # print('XspediusTraffic matrices dims of eval: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class SwitchL3Topology(object):  # **#需更改类名称

    def __init__(self, config, data_dir='./data/'):
        self.DG = nx.DiGraph()

        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'

        self.data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        topology_file_dir = self.data_dir + 'topology_file'
        topology_file_xml = topology_file_dir + '/' + 'topology-zoo.org_files_SwitchL3.xml'  # **# 需更改.xml文件名

        # 解析 XML 文件
        tree = ET.parse(topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='id') != None:
                # for g_tag in graph_tag.iter(tag=None):
                node_id = graph_tag.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                self.node_id_list.append(node_id)
                self.node_id_to_idx[node_id] = index
                index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        assert root[-1].get(key='edgedefault') == "undirected", "ERROE，请检查.xml文件的graph位置"
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='source') != None:
                source_node_id = graph_tag.get(key='source')
                target_node_id = graph_tag.get(key='target')
                if (source_node_id, target_node_id) in self.link_id_list:
                    continue
                self.link_id_list.append((source_node_id, target_node_id))
                self.link_id_list.append((target_node_id, source_node_id))  # 注意我们考虑的是单向link，所以zoo数据集中的每个link应该看成两条
                self.link_idx_list.append((self.node_id_to_idx[source_node_id], self.node_id_to_idx[target_node_id]))
                self.link_idx_list.append(
                    (self.node_id_to_idx[target_node_id], self.node_id_to_idx[source_node_id]))  # 同理

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class SwitchL3Traffic(object):  # **#需更改类名称

    def __init__(self, config, topology, data_dir='./data/', is_training=True):
        self.tm_save_file = data_dir + config.topology_file + '/' + config.topology_file + '_TM.pkl'
        if is_training:
            self.traffic_file = data_dir + config.topology_file + '/' + config.traffic_file  #
        else:
            self.traffic_file = data_dir + config.topology_file + '/' + config.test_traffic_file  #
        self.num_nodes = topology.num_nodes
        self.load_traffic(topology, config, is_training)

    def load_traffic(self, topology, config, is_training):

        print('[*] Generate traffic matrices...')

        if os.path.exists(self.tm_save_file):
            with open(self.tm_save_file, 'rb') as pkl:
                self.tm = pickle.load(pkl)
            # assert len(self.optimal_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
        else:
            # tm = tmgen.models.modulated_gravity_tm()
            print("TM模拟")
            mean_flow_traffic = 10000
            self.tm = modulated_gravity_tm(num_nodes=self.num_nodes,  #
                                           num_tms=672,  # 4*24*7 # 12*24*7=2016
                                           mean_traffic=mean_flow_traffic,  # TM中flow平均值的最小值
                                           pm_ratio=1.5,
                                           t_ratio=.75,
                                           diurnal_freq=6 / (24 * 24),  # 1 / 24,
                                           spatial_variance=mean_flow_traffic * 0.5,  # 单个TM内部变化
                                           temporal_variance=50)  # 100

            with open(self.tm_save_file, 'wb') as handle:
                pickle.dump(self.tm, handle)

        if is_training:
            self.traffic_matrices = self.tm#[:96*5]##训练
        else:
            self.traffic_matrices = self.tm[96*5:96*7]##测试

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]  # 注意它的设置
        print('XspediusTraffic matrices dims of training: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))

        # tms_shape = self.traffic_matrices_eval.shape
        # self.tm_cnt_eval = tms_shape[0]#注意它的设置
        # print('XspediusTraffic matrices dims of eval: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class UunetTopology(object):  # **#需更改类名称

    def __init__(self, config, data_dir='./data/'):
        self.DG = nx.DiGraph()

        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'

        self.data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        topology_file_dir = self.data_dir + 'topology_file'
        topology_file_xml = topology_file_dir + '/' + 'topology-zoo.org_files_Uunet.xml'  # **# 需更改.xml文件名

        # 解析 XML 文件
        tree = ET.parse(topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='id') != None:
                # for g_tag in graph_tag.iter(tag=None):
                node_id = graph_tag.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                self.node_id_list.append(node_id)
                self.node_id_to_idx[node_id] = index
                index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        assert root[-1].get(key='edgedefault') == "undirected", "ERROE，请检查.xml文件的graph位置"
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='source') != None:
                source_node_id = graph_tag.get(key='source')
                target_node_id = graph_tag.get(key='target')
                if (source_node_id, target_node_id) in self.link_id_list:
                    continue
                self.link_id_list.append((source_node_id, target_node_id))
                self.link_id_list.append((target_node_id, source_node_id))  # 注意我们考虑的是单向link，所以zoo数据集中的每个link应该看成两条
                self.link_idx_list.append((self.node_id_to_idx[source_node_id], self.node_id_to_idx[target_node_id]))
                self.link_idx_list.append(
                    (self.node_id_to_idx[target_node_id], self.node_id_to_idx[source_node_id]))  # 同理

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class UunetTraffic(object):  # **#需更改类名称

    def __init__(self, config, topology, data_dir='./data/', is_training=True):
        self.tm_save_file = data_dir + config.topology_file + '/' + config.topology_file + '_TM.pkl'
        if is_training:
            self.traffic_file = data_dir + config.topology_file + '/' + config.traffic_file  #
        else:
            self.traffic_file = data_dir + config.topology_file + '/' + config.test_traffic_file  #
        self.num_nodes = topology.num_nodes
        self.load_traffic(topology, config, is_training)

    def load_traffic(self, topology, config, is_training):

        print('[*] Generate traffic matrices...')

        if os.path.exists(self.tm_save_file):
            with open(self.tm_save_file, 'rb') as pkl:
                self.tm = pickle.load(pkl)
            # assert len(self.optimal_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
        else:
            # tm = tmgen.models.modulated_gravity_tm()
            print("TM模拟")
            mean_flow_traffic = 10000
            self.tm = modulated_gravity_tm(num_nodes=self.num_nodes,  #
                                           num_tms=672,  # 4*24*7 # 12*24*7=2016
                                           mean_traffic=mean_flow_traffic,  # TM中flow平均值的最小值
                                           pm_ratio=1.5,
                                           t_ratio=.75,
                                           diurnal_freq=6 / (24 * 24),  # 1 / 24,
                                           spatial_variance=mean_flow_traffic * 0.5,  # 单个TM内部变化
                                           temporal_variance=50)  # 100

            with open(self.tm_save_file, 'wb') as handle:
                pickle.dump(self.tm, handle)

        if is_training:
            self.traffic_matrices = self.tm#[:96*5]##训练
        else:
            self.traffic_matrices = self.tm[96*5:96*7]##测试

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]  # 注意它的设置
        print('XspediusTraffic matrices dims of training: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))

        # tms_shape = self.traffic_matrices_eval.shape
        # self.tm_cnt_eval = tms_shape[0]#注意它的设置
        # print('XspediusTraffic matrices dims of eval: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class CernetTopology(object):  # **#需更改类名称

    def __init__(self, config, data_dir='./data/'):
        self.DG = nx.DiGraph()

        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'

        self.data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        topology_file_dir = self.data_dir + 'topology_file'
        topology_file_xml = topology_file_dir + '/' + 'topology-zoo.org_files_Cernet.xml'  # **# 需更改.xml文件名

        # 解析 XML 文件
        tree = ET.parse(topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='id') != None:
                # for g_tag in graph_tag.iter(tag=None):
                node_id = graph_tag.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                self.node_id_list.append(node_id)
                self.node_id_to_idx[node_id] = index
                index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        assert root[-1].get(key='edgedefault') == "undirected", "ERROE，请检查.xml文件的graph位置"
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='source') != None:
                source_node_id = graph_tag.get(key='source')
                target_node_id = graph_tag.get(key='target')
                if (source_node_id, target_node_id) in self.link_id_list:
                    continue
                self.link_id_list.append((source_node_id, target_node_id))
                self.link_id_list.append((target_node_id, source_node_id))  # 注意我们考虑的是单向link，所以zoo数据集中的每个link应该看成两条
                self.link_idx_list.append((self.node_id_to_idx[source_node_id], self.node_id_to_idx[target_node_id]))
                self.link_idx_list.append(
                    (self.node_id_to_idx[target_node_id], self.node_id_to_idx[source_node_id]))  # 同理

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class DfnTopology(object):  # **#需更改类名称

    def __init__(self, config, data_dir='./data/'):
        self.DG = nx.DiGraph()

        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'

        self.data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        topology_file_dir = self.data_dir + 'topology_file'
        topology_file_xml = topology_file_dir + '/' + 'topology-zoo.org_files_Dfn.xml'  # **# 需更改.xml文件名

        # 解析 XML 文件
        tree = ET.parse(topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='id') != None:
                # for g_tag in graph_tag.iter(tag=None):
                node_id = graph_tag.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                self.node_id_list.append(node_id)
                self.node_id_to_idx[node_id] = index
                index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        assert root[-1].get(key='edgedefault') == "undirected", "ERROE，请检查.xml文件的graph位置"
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='source') != None:
                source_node_id = graph_tag.get(key='source')
                target_node_id = graph_tag.get(key='target')
                if (source_node_id, target_node_id) in self.link_id_list:
                    continue
                self.link_id_list.append((source_node_id, target_node_id))
                self.link_id_list.append((target_node_id, source_node_id))  # 注意我们考虑的是单向link，所以zoo数据集中的每个link应该看成两条
                self.link_idx_list.append((self.node_id_to_idx[source_node_id], self.node_id_to_idx[target_node_id]))
                self.link_idx_list.append(
                    (self.node_id_to_idx[target_node_id], self.node_id_to_idx[source_node_id]))  # 同理

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class DfnTraffic(object):  # **#需更改类名称

    def __init__(self, config, topology, data_dir='./data/', is_training=True):
        self.tm_save_file = data_dir + config.topology_file + '/' + config.topology_file + '_TM.pkl'
        if is_training:
            self.traffic_file = data_dir + config.topology_file + '/' + config.traffic_file  #
        else:
            self.traffic_file = data_dir + config.topology_file + '/' + config.test_traffic_file  #
        self.num_nodes = topology.num_nodes
        self.load_traffic(topology, config, is_training)

    def load_traffic(self, topology, config, is_training):

        print('[*] Generate traffic matrices...')

        if os.path.exists(self.tm_save_file):
            with open(self.tm_save_file, 'rb') as pkl:
                self.tm = pickle.load(pkl)
            # assert len(self.optimal_mlu_save) == self.tm_cnt, "Error from the self.optimal_mlu_save"
        else:
            # tm = tmgen.models.modulated_gravity_tm()
            print("TM模拟")
            mean_flow_traffic = 10000
            self.tm = modulated_gravity_tm(num_nodes=self.num_nodes,  #
                                           num_tms=672,  # 4*24*7 # 12*24*7=2016
                                           mean_traffic=mean_flow_traffic,  # TM中flow平均值的最小值
                                           pm_ratio=1.5,
                                           t_ratio=.75,
                                           diurnal_freq=6 / (24 * 24),  # 1 / 24,
                                           spatial_variance=mean_flow_traffic * 0.5,  # 单个TM内部变化
                                           temporal_variance=50)  # 100

            with open(self.tm_save_file, 'wb') as handle:
                pickle.dump(self.tm, handle)

        if is_training:
            self.traffic_matrices = self.tm#[:96*5]##训练
        else:
            self.traffic_matrices = self.tm[96*5:96*7]##测试

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]  # 注意它的设置
        print('XspediusTraffic matrices dims of training: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))

        # tms_shape = self.traffic_matrices_eval.shape
        # self.tm_cnt_eval = tms_shape[0]#注意它的设置
        # print('XspediusTraffic matrices dims of eval: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class ChinanetTopology(object):  # **#需更改类名称

    def __init__(self, config, data_dir='./data/'):
        self.DG = nx.DiGraph()

        self.topology_file = data_dir + config.topology_file + '/' + config.topology_file  #
        self.shortest_paths_file = self.topology_file + '_shortest_paths'

        self.data_dir = data_dir + config.topology_file + '/'  #

        self.generate_topology_file()  #

        self.load_topology()
        self.calculate_paths()

    def generate_topology_file(self):  ##

        topology_file_dir = self.data_dir + 'topology_file'
        topology_file_xml = topology_file_dir + '/' + 'topology-zoo.org_files_Chinanet.xml'  # **# 需更改.xml文件名

        # 解析 XML 文件
        tree = ET.parse(topology_file_xml)
        root = tree.getroot()

        # 取出node信息，将 id 转为 index，例 id:12 => index:0, id:13 => index:1, ..., id:15 => idex:22
        self.node_id_to_idx = {}
        self.node_id_list = []
        index = 0
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='id') != None:
                # for g_tag in graph_tag.iter(tag=None):
                node_id = graph_tag.get(key='id')  # get(key, default=None) 获取元素指定key对应的attrib，如果没有attrib，返回default。
                self.node_id_list.append(node_id)
                self.node_id_to_idx[node_id] = index
                index += 1
        num_nodes = len(self.node_id_list)
        # index 转 id
        self.node_idx_to_id = self.node_id_list

        # 取出link信息,并重新排序，例：[(0, 10), (0, 11), (0, 13), (1, 2), (1, 13), (1, 16), (1, 20), (2, 1), ...]
        self.link_id_list = []
        self.link_idx_list = []
        assert root[-1].get(key='edgedefault') == "undirected", "ERROE，请检查.xml文件的graph位置"
        for graph_tag in root[-1].iter(tag=None):  # iter(tag=None) 遍历Element的child，可以指定tag精确查找
            if graph_tag.get(key='source') != None:
                source_node_id = graph_tag.get(key='source')
                target_node_id = graph_tag.get(key='target')
                if (source_node_id, target_node_id) in self.link_id_list:
                    continue
                self.link_id_list.append((source_node_id, target_node_id))
                self.link_id_list.append((target_node_id, source_node_id))  # 注意我们考虑的是单向link，所以zoo数据集中的每个link应该看成两条
                self.link_idx_list.append((self.node_id_to_idx[source_node_id], self.node_id_to_idx[target_node_id]))
                self.link_idx_list.append(
                    (self.node_id_to_idx[target_node_id], self.node_id_to_idx[source_node_id]))  # 同理

        link_idx_sort_list = []
        for i_l in range(len(self.link_idx_list)):
            heapq.heappush(link_idx_sort_list,
                           (self.link_idx_list[i_l][0], self.link_idx_list[i_l][1], self.link_idx_list[i_l]))
        self.link_idx_sort_list = [heapq.heappop(link_idx_sort_list)[-1] for i_l in range(len(self.link_idx_list))]
        num_links = len(self.link_idx_sort_list)

        # 写入数据到拓扑文件
        if not os.path.exists(self.topology_file):  #

            with open(self.topology_file, 'w') as f:
                f.write('Node_num: {}\tEdge_num: {}\n'.format(num_nodes, num_links))
                f.write('Link_index\tSource\tDestination\tWeights\tCapacity(kbps)\tNode_id_pair\n')
                for idx in range(num_links):
                    f.write("{:d}\t{:d}\t{:d}\t1\t10000000\t('{:s}', '{:s}')\n".format(idx,
                                                                                       self.link_idx_sort_list[idx][0],
                                                                                       self.link_idx_sort_list[idx][1],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               0]],
                                                                                       self.node_id_list[
                                                                                           self.link_idx_sort_list[idx][
                                                                                               1]]))

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c, n_id_p = link  #
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = float(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), float(w))], weight="weight")

        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        # nx.draw_networkx(self.DG,arrows=None, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()
        #
        # nx.draw(self.DG, with_labels=True)
        # plt.title('The network topology of GEANT')
        # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(
                        node_path).size  # np.unique用于返回数组中的唯一值，并按照指定的排序方式进行排序，两者相等说明路径中没有重复值（即没有环路）
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(
                            list(nx.all_shortest_paths(self.DG, s, d, weight='weight', method='dijkstra'))[
                            :MAX_PATH_NUM])
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class NetworkEnvironment(object):

    def __init__(self, config, is_training=True):
        self.data_dir = './data/'
        self.topology = AbileneTopology(config, self.data_dir)
        self.traffic = AbileneTraffic(config, self.topology.num_nodes, self.data_dir, is_training=is_training)
        self.traffic_matrices = self.traffic.traffic_matrices * 100 * 8 / 300 / 1000  # 转为单位kbps ## Unit: (100 bytes / 5 minutes)  // 100 is the pkt sampling rate
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
        self.shortest_paths_link = self.convert_to_edge_path(self.shortest_paths_node)  # paths consist of links

    def convert_to_edge_path(self, node_paths):
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


# cimport numpy
# import numpy
from six.moves import range
from tmgen.exceptions import TMgenException


# from tmgen.tm cimport TrafficMatrix

def peak_mean_cycle(freq, n, mean, peak_to_mean, trough_to_mean=np.nan):
    """
    Generate a simple sinusoid with specified frequency, length, mean,
    peak-to-mean ratio, and trough_to_mean ratio.

    The generated signal has the form
    :math:`x = mean*(peaktomean-1)*sin(2*\\pi*freq*linspace(0,1,N))+mean`

    Note that if the mean m is 0, then
    :math:`x = peakmean*sin(2*\\pi*freq*linspace(0,1,N))`

    Intervals of x below the mean have a drawdown determined by trough_to_mean.

    ..note
        the trough-to-mean ratio must be less than or equal to the
        peak-to-mean ratio.

    :param freq: frequency
    :param n: number of samples
    :param mean: mean value
    :param peak_to_mean: ratio of peak to mean. Must be >= 1 (1 resulting
        in a constant, flatlined signal)
    :param trough_to_mean: ratio of through to mean
    :return: numpy array containing the y-values for the sinusoid
    """
    # sanity checks
    if peak_to_mean < 1:
        raise TMgenException(
            'Peak-to-mean ratio must be greater than or equal to 1')
    # Generate the basic sinusoid
    base = np.sin(
        2 * np.pi * freq * np.linspace(0, n - 1, n))
    # If mean is 0, simply return scaled sinusoid
    if mean == 0:
        return peak_to_mean * base
    # cdef np.ndarray x, y
    # cdef int i
    x = mean * (peak_to_mean - 1) * base + mean
    if not np.isnan(trough_to_mean):
        if trough_to_mean > peak_to_mean:
            raise TMgenException(
                'Trough-to-mean ratio must be less than or equal to '
                'peak-to-mean ratio')
        if trough_to_mean == 1:
            return x  # no effect on trough
        y = mean * (trough_to_mean - 1) * base + mean
        x[x < mean] = y[x < mean]
    return x


def modulated_gravity_tm(num_nodes, num_tms, mean_traffic,
                         pm_ratio=1.5,
                         t_ratio=.75,
                         diurnal_freq=1 / 24,
                         spatial_variance=100,
                         temporal_variance=0.01):
    """
    Generate a modulated gravity traffic matrix with the given parameters

    :param num_nodes: number of Points-of-Presence (i.e., origin-destination pairs)
    :param num_tms: total number of traffic matrices to generate (i.e., time epochs)
    :param mean_traffic: the average total volume of traffic#实际上是单个TM的总流量的最小值
    ##更改为TM中flow的流量均值的最小值
    :param pm_ratio: peak-to-mean ratio. Peak traffic will be larger by
        this much (must be bigger than 1). Default is 1.5
    :param t_ratio: trough-to-mean ratio. Default is 0.75
    :param diurnal_freq: Frequency of modulation. Default is 1/24 (i.e., hourly)
        if you are generating multi-day TMs
    :param spatial_variance: Variance on the volume of traffic between
        origin-destination pairs.
        Pick something reasonable with respect to your mean_traffic.
        Default is 100
    :param temporal_variance: Variance on the volume in time
    :return:
    """

    # generate total traffic
    sinusoid = peak_mean_cycle(diurnal_freq, num_tms,
                               mean_traffic, pm_ratio,
                               t_ratio)
    ###
    spatial_random = np.random.normal(loc=0, scale=temporal_variance, size=num_tms)
    sinusoid = sinusoid + spatial_random  # sinusoid增加了空间上的基于”spatial_variance“的随机变化值
    ###

    if np.min(sinusoid) < 0:
        # rescale
        sinusoid = sinusoid + np.absolute(np.min(sinusoid))
        mean_traffic = np.mean(sinusoid)

    base_random_gravity_tm = random_gravity_tm(num_nodes,
                                               num_nodes * num_nodes,
                                               spatial_variance,
                                               num_tms)
    sinusoid_repeated = np.repeat(sinusoid[:, np.newaxis, np.newaxis], repeats=num_nodes, axis=1)
    sinusoid_repeated = np.repeat(sinusoid_repeated, repeats=num_nodes, axis=2)
    tm = base_random_gravity_tm * sinusoid_repeated

    rn_mean = 0
    rn_variance = spatial_variance
    add_rn = np.random.normal(loc=rn_mean, scale=rn_variance, size=(num_tms, num_nodes, num_nodes))

    tm_add_rn = (tm + add_rn).clip(min=0)

    plt.plot(tm_add_rn[0].reshape(-1))
    plt.title('tm.at_time(0)')
    plt.show()
    print(tm_add_rn[0].sum())

    return tm_add_rn  # tm_add_rn#tm

    # # generate total traffic
    # cdef numpy.ndarray sinusoid = _peak_mean_cycle(diurnal_freq, num_tms,
    #                                                mean_traffic, pm_ratio,
    #                                                t_ratio)
    # if numpy.min(sinusoid) < 0:
    #     # rescale
    #     sinusoid = sinusoid + numpy.absolute(numpy.min(sinusoid))
    #     mean_traffic = numpy.mean(sinusoid)
    #
    # base_random_gravity_tm = random_gravity_tm(num_nodes,
    #                                            mean_traffic / num_nodes)
    # tm = numpy.concatenate(
    #     [base_random_gravity_tm.matrix * x for x in sinusoid], axis=2)
    # return TrafficMatrix(tm)


def random_gravity_tm(num_nodes, mean_traffic, spatial_variance, num_tms):
    """
    Random gravity model, parametrized by the mean traffic per ingress-egress
    pair.
    See http://dl.acm.org/citation.cfm?id=1096551 for full description

    :param num_nodes: number of nodes in the network
    :param mean_traffic: average traffic volume between a pair of nodes
    :return: a new :py:class:`~TrafficMatrix`
    """

    # dist = np.random.rand(num_nodes, 1)
    # dist = dist / sum(dist)
    # matrix = np.matmul(dist, dist.T).clip(min=0) * mean_traffic
    # return matrix.reshape((num_nodes, num_nodes, 1))

    ###

    ###**# 1. abs方案 #
    # mean = 1
    # dist = np.random.normal(loc=mean, scale=spatial_variance, size=(num_tms, num_nodes, 1))

    # #dist = dist / (dist.sum(axis=1)[:, None])
    # dist = dist / (np.ones(shape=(num_tms, 1, 1)) * num_nodes * mean)#用此句代码代替上一句代码
    #
    # matrix = abs( dist @ np.transpose(dist, (0, 2, 1)))

    # ###**# 2. clip方案 #
    # mean = 1#spatial_variance=2
    # dist = np.random.normal(loc=mean, scale=spatial_variance, size=(num_tms, num_nodes, 1))
    # # dist = dist / (dist.sum(axis=1)[:, None])
    # dist = dist / (np.ones(shape=(num_tms, 1, 1)) * num_nodes * mean)#用此句代码代替上一句代码
    #
    # # dist = dist.clip(min=0)
    # matrix = dist @ np.transpose(dist, (0, 2, 1))
    # matrix = matrix.clip(min=0)
    #
    # plt.plot(matrix[1].reshape(-1))
    # plt.title('tm.at_time(1)')
    # plt.show()
    # print(matrix[1].sum())

    ###**# 3. 叠加方案，先固定随机生成的TM，以此为基础，再加上不同方差参数的随机数值

    seed = 2  # 5 #12 # 2 #16 #9 #5 #**#Xspedius：2; SwitchL3: 12; Uunet：
    np.random.seed(seed)
    # random.seed(seed)

    mean = 1  #
    variance = 3
    # dist = np.random.normal(loc=mean, scale=spatial_variance, size=(num_tms, num_nodes, 1))
    # # dist = dist / (dist.sum(axis=1)[:, None])
    # dist = dist / (np.ones(shape=(num_tms, 1, 1)) * num_nodes * mean)#用此句代码代替上一句代码
    #
    dist = np.random.normal(loc=mean, scale=variance, size=(num_nodes, 1))
    dist = np.expand_dims(dist, axis=0)
    dist = np.repeat(dist, num_tms, axis=0)

    # dist = dist / (dist.sum(axis=1)[:, None])
    dist = dist / (np.ones(shape=(num_tms, 1, 1)) * num_nodes * mean)  # 用此句代码代替上一句代码

    # dist = dist.clip(min=0)
    matrix = dist @ np.transpose(dist, (0, 2, 1))  # 使用 @ 运算符进行矩阵相乘
    matrix = matrix.clip(min=0) * mean_traffic

    plt.plot(matrix[1].reshape(-1))
    plt.title('tm.000')
    plt.show()
    print(matrix[1].sum())

    #
    # rn_mean = 0
    # rn_variance = spatial_variance
    # add_rn= mean_traffic * 0.01 * \
    #         np.random.normal(loc=rn_mean, scale=rn_variance, size=(num_tms, num_nodes, num_nodes))
    #
    # matrix_add_rn = (matrix + add_rn).clip(min=0)
    # plt.plot(matrix_add_rn[1].reshape(-1))
    # plt.title('tm.at_time(1)')
    # plt.show()
    # print(matrix_add_rn[1].sum())

    return matrix
    ###
