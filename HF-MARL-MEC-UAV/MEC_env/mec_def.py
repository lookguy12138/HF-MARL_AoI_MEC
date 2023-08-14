# -*- coding: UTF-8 -*-
import numpy as np
import random
import logging
import math

logging.basicConfig(level=logging.WARNING)

# UAV动作
class Action(object):
    def __init__(self):
        self.move = None        # 移动
        self.collect = None     # 收集？？
        self.offloading = []    # 卸载
        self.bandwidth = 0      # 带宽
        self.execution = []     # 计算


# UAV状态
class AgentState(object):
    def __init__(self):
        self.position = None    # 位置
        self.obs = None         # 观察


# 边缘设备
class EdgeDevice(object):
    edge_count = 0  # 总数

    def __init__(self, obs_r, pos, spd, collect_r, max_buffer_size, MAX_EP_STEPS, movable=True, mv_bt=0, trans_bt=0):  # pos(x,y,h)
        self.no = EdgeDevice.edge_count  # 标号
        EdgeDevice.edge_count += 1  # 总数+1
        self.obs_r = obs_r  # 观察半径
        self.init_pos = pos  # 初始位置
        self.position = pos  # 当前位置
        self.move_r = spd  # 移动速率
        self.collect_r = collect_r  # 收集速率
        self.mv_battery_cost = mv_bt  # 移动能耗
        self.trans_battery_cost = trans_bt  # 通信能耗
        self.data_buffer = {}  # 数据缓冲区
        self.max_buffer_size = max_buffer_size  # 数据缓冲区上限
        self.idle = True  # 收集空闲
        self.movable = movable  # 可移动性
        self.state = AgentState()  # UAV状态
        self.action = Action()  # UAV动作
        self.offloading_idle = True  # 卸载空闲
        self.total_data = {}  # 待计算数据
        self.computing_rate = 2e4  # 计算速率 2e4
        self.computing_idle = True  # 计算空闲
        self.index_dim = 2  # 索引维度？？
        self.collecting_sensors = []  # UAV正在从中收集数据的传感器的编号
        self.ptr = 0.2  # UAV-center 最大传输功率
        self.h = 100  # 飞行高度 5——》100  # ***
        self.noise = 1e-13  # 噪声功率谱密度
        self.trans_rate = 0  # 通信速率

        self.t_distance = 0  # UAV在本时隙的累计移动距离  # ***
        self.e_distance = 0  # UAV在本episode的累计移动距离  # ***
        self.total_distance = 0  # UAV在本episode的移动距离上限  # ***

    # UAV移动
    def move(self, new_move, h):
        # 若 收集空闲
        if self.idle:
            self.position += new_move  # 位置更新
            self.mv_battery_cost += np.linalg.norm(new_move)  # 移动能耗更新

    # （在MAAC_agent中使用）获取待计算数据（将total_data中的待计算数据存储在total_data_state中）
    # 输出：待计算数据列表 shape为(2, 缓冲区上限) 0->数据量 1->age
    def get_total_data(self):
        total_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        # print("total_data", self.total_data)
        if self.total_data:
            for j, k in enumerate(list(self.total_data.keys())):
                # print(self.total_data[k])
                total_data_state[0, j] = self.total_data[k][0]
                total_data_state[1, j] = self.total_data[k][1]
        return total_data_state

    # （在MAAC_agent中使用）获取待计算数据（将total_data中的待计算数据存储在total_data_state中）
    # 输出：计算完成数据列表 shape为(2, 缓冲区上限) 0->数据量 1->age
    # ***************************以后把这个函数删了，统一都用get_total_data
    def get_done_data(self):
        total_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        # print("total_data", self.total_data)
        if self.total_data:
            for j, k in enumerate(list(self.total_data.keys())):
                # print(self.total_data[k])
                total_data_state[0, j] = self.total_data[k][0]
                total_data_state[1, j] = self.total_data[k][1]
        return total_data_state

    # 数据更新（这个函数没有被用过！）
    def data_update(self, pak):
        # 若pak[1]在data_buffer的key中
        print("pak", pak)
        if pak[1] in self.data_buffer.keys():
            self.data_buffer[pak[1]].append(pak)
        else:
            self.data_buffer[pak[1]] = [pak]

    # one-hot 本地计算
    def process(self, t=1):  # one-hot local execution
        # 若UAV没有 待计算数据
        if not self.total_data:
            return False, {}
        # age更新
        for k in self.total_data.keys():
            self.total_data[k][1] += t
        # 过程
        # 若 UAV有待计算数据 且 动作-计算为真
        if self.total_data and sum(self.action.execution):
            # 提取待计算数据缓冲区（字典）中的所有 (key,value)
            data2process = [[k, d] for k, d in self.total_data.items()]
            # 若 动作-计算 的索引超出了 待计算数据 的上限
            # 则将 动作-计算 的索引修正为 随机一个有效索引
            if self.action.execution.index(1) >= len(data2process):
                self.action.execution[self.action.execution.index(1)] = 0
                self.action.execution[np.random.randint(len(data2process))] = 1
                # print(self.action.execution)
            # 计算空闲设定为否
            self.computing_idle = False
            # 更新待计算数据
            self.total_data[data2process[self.action.execution.index(1)][0]][0] -= self.computing_rate * t
            # 若计算完成
            if self.total_data[data2process[self.action.execution.index(1)][0]][0] <= 0:
                # 记录传感器编号
                sensor_index = data2process[self.action.execution.index(1)][0]
                # 记录packet的AoI
                sensor_aoi = self.total_data[data2process[self.action.execution.index(1)][0]][1]
                # 记录packet的数据量
                sensor_data = self.data_buffer[sensor_index][0]
                # 在UAV缓冲区中删除这个packet
                del self.data_buffer[sensor_index]
                # 在 待计算数据 中删除计算完的数据
                self.total_data.pop(data2process[self.action.execution.index(1)][0])
                # 计算空闲设定为真
                self.computing_idle = True
                return True, {sensor_index: [sensor_data, sensor_aoi]}
        return False, {}


# 没用
def agent_com(agent_list: object) -> object:
    age_dict = {}
    # 对 所有UAV agent
    for u in agent_list:
        # 对 数据缓冲区
        for k, v in u.data_buffer.items():
            if k not in age_dict:
                age_dict[k] = v[-1][1]
            elif age_dict[k] > v[-1][1]:
                age_dict[k] = v[-1][1]
    return age_dict

# sensor动作
class sensor_Action(object):
    def __init__(self):
        self.computing = []  # 本地计算

# 地面sensor
class Sensor(object):
    sensor_cnt = 0  # 总数

    def __init__(self, pos, data_rate, bandwidth, max_ds, max_buffer_size, lam=0.5, weight=1):
        self.no = Sensor.sensor_cnt  # 标号
        Sensor.sensor_cnt += 1  # 总数+1
        self.position = pos  # 位置（不可移动）
        self.weight = weight
        self.data_rate = data_rate  # 数据生成速率
        self.bandwidth = bandwidth  # 带宽
        self.trans_rate = 8e3  # 传输速率 8e3
        self.data_buffer = []  # 数据缓冲区
        self.max_data_size = max_ds  # 数据大小上限
        self.data_state = bool(len(self.data_buffer))  # 数据缓冲区中有数据
        self.collect_state = False  # 收集状态
        self.lam = lam  # Poisson分布中，间隔时间的期望（用于数据生成）
        self.noise_power = 1e-13 * self.bandwidth  # 噪声功率（和带宽呈正比）
        self.gen_threshold = 0.3  # 30%的概率不生成待计算数据

        self.computing_rate = 2e2  # sensor计算速率 2e2
        self.computing_state = False  # sensor计算状态
        self.computing_time = 0  # 计算时间
        self.total_data = [0, 0]  # sensor本地总计算数据
        self.action = sensor_Action()  # UAV动作
        self.index_dim = 2
        self.max_buffer_size = max_buffer_size * 2  # 数据缓冲区上限

    def data_gen(self, t=1):
        # sensor生成的待计算数据
        new_data = self.data_rate * np.random.poisson(self.lam)
        # new_data = min(new_data, self.max_data_size)
        # 若 生成待计算数据>数据大小上限 or 随机到本时隙不生成待计算数据 or 源缓冲区已满（上限为10）
        if new_data >= self.max_data_size or random.random() >= self.gen_threshold or len(self.data_buffer) >= self.max_buffer_size:
            return
        # 将sensor新生成的待计算数据，放入数据缓冲区
        if new_data:
            self.data_buffer.append([new_data, 0, self.no])
            self.data_state = True
        # age更新
        if self.data_buffer:
            for i in range(len(self.data_buffer)):
                self.data_buffer[i][1] += t

    # （在MAAC_agent中使用）获取sensor待计算数据（将data_buffer中的数据存储在sensor_data_state中）
    # 输出：sensor待计算数据列表 shape为(2, 缓冲区上限) 0->数据量 1->age
    def get_sensor_data(self):
        sensor_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        if self.data_buffer:
            for m, k in enumerate(self.data_buffer):
                sensor_data_state[0, m] = k[0]
                sensor_data_state[1, m] = k[1]
        return sensor_data_state

    def process(self, t=1):
        # sensor总数据量
        total_size = 0
        # 若①sensor计算空闲 and ②sensor源缓冲区中有数据
        if not self.computing_state and self.data_buffer:
            # 收集sensor源缓冲区中部分数据，用于本地计算
            for data in self.data_buffer:
                total_size += data[0]
            tmp_size = int(total_size * self.action.computing[0])  # 计划进行计算的量
            if tmp_size == 0:
                return False, {}
            tmp2_size = tmp_size
            # sensor处于本地计算状态
            self.computing_state = True
            # 从sensor源计算缓冲区计算数据的age
            tmp_age = 0
            # 从sensor源计算缓冲区计算数据的标号
            tmp_index = 0
            # 从sensor源计算缓冲区计算数据的标号
            for i, data in enumerate(self.data_buffer):
                tmp2_size = tmp2_size - data[0]
                if tmp2_size <= 0:
                    tmp_index = i
                    break
            tmp2_size = tmp2_size + self.data_buffer[tmp_index][0]
            # 最小age
            # tmp_age = self.data_buffer[tmp_index][1]
            # 加权平均age
            for i in range(tmp_index):
                tmp_age = tmp_age + self.data_buffer[i][1] * (self.data_buffer[i][0] / tmp_size)
            tmp_age = tmp_age + self.data_buffer[tmp_index][1] * (tmp2_size / tmp_size)

            # sensor本地计算所需的时间
            self.computing_time = tmp_size / self.computing_rate
            # 将从“sensor源缓冲区”中收集到的数据存储到“sensor待计算缓冲区”
            self.total_data = [tmp_size, tmp_age + self.computing_time - 1]
            # 删除sensor源缓冲区中用于本地计算的数据
            self.data_buffer[tmp_index][0] -= tmp2_size
            if self.data_buffer[tmp_index][0] == 0:
                del self.data_buffer[tmp_index]
            for i in range(tmp_index):
                del self.data_buffer[0]
            return False, {}

        # 若 sensor正在进行本地计算
        elif self.computing_state:
            # sensor本地计算时间-1
            self.computing_time -= 1
            # 当待计算数据全部被计算完成
            if self.computing_time <= 0:
                # 记录传感器编号
                sensor_index = self.no
                # 记录packet的AoI
                sensor_aoi = self.total_data[1]
                # 记录packet的数据量
                sensor_data = self.total_data[0]
                # sensor处于计算空闲状态
                self.computing_state = False
                # sensor本地计算时间
                self.computing_time = 0
                # 在sensor待计算缓冲区中删除
                self.total_data = [0, 0]
                return True, {sensor_index: [sensor_data, sensor_aoi]}
            return False, {}
        # sensor源数据缓冲区中没数据
        else:
            return False, {}


# 收集信道参数
collecting_channel_param = {'suburban': (4.88, 0.43, 0.1, 21),
                            'urban': (9.61, 0.16, 1, 20),
                            'dense-urban': (12.08, 0.11, 1.6, 23),
                            'high-rise-urban': (27.23, 0.08, 2.3, 34)}
# 收集参数：四选一（郊区、城市、密集城市、高层城市）
collecting_params = collecting_channel_param['urban']
a = collecting_params[0]
b = collecting_params[1]
yita0 = collecting_params[2]
yita1 = collecting_params[3]
carrier_f = 2.4e9  # ***


# 收集速率
def collecting_rate(sensor, agent):
    # sensor和UAV之间的欧氏距离
    d = np.linalg.norm(np.array(sensor.position) - np.array(agent.position))
    # 根据公式计算收集速率
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / d) - a)))
    L = Pl * yita0 + yita1 * (1 - Pl)
    gamma = agent.ptr_col / (L * sensor.noise_power**2)
    rate = sensor.bandwidth * np.log2(1 + gamma)
    return rate


# 待计算数据收集
# 输入（各个sensor，UAV，UAV悬停时间）
# 输出 UAV悬停时间
def data_collecting(sensors, agent, hovering_time):
    # 进行收集
    # 若①收集空闲 and ②UAV待计算数据数量<最大缓冲区长度
    if agent.idle and (len(agent.total_data.keys()) < agent.max_buffer_size):
        # obs_sensor = []
        # 数据属性（所有sensor的收集时间列表）
        data_properties = []

        # 对 每个sensor 循环
        for sensor in sensors:
            # 若sensor的数据缓冲区没有数据
            if not sensor.data_buffer:
                continue
            # 若①sensor在UAV覆盖范围内；②sensor不处于收集状态；③sensor标号不在UAV待计算数据的key值中
            if (np.linalg.norm(np.array(sensor.position) - np.array(agent.position)) <= agent.collect_r) and not(sensor.collect_state) and not(sensor.no in agent.total_data.keys()):
                # sensor处于被收集状态
                sensor.collect_state = True
                agent.collecting_sensors.append(sensor.no)
                # UAV处于收集忙碌状态
                agent.idle = False
                # 如果UAV待计算数据数量 >= 缓冲区
                if len(agent.total_data.keys()) >= agent.max_buffer_size:
                    continue

                # sensor总数据量
                total_size = 0
                # UAV从sensor收集的数据量
                tmp_size = 0
                # UAV从sensor收集数据的age
                tmp_age = 0
                # UAV从sensor收集到数据的标号
                tmp_index = 0

                # 计算收集源缓冲区中的数据量
                for data in sensor.data_buffer:
                    total_size += data[0]
                # 收集源缓冲区中最老的XX%数据（XX%由神经网络决定）
                # tmp_size = int(total_size * sensor.action.collecting[0])
                # 收集源缓冲区中所有数据
                tmp_size = int(total_size * 1)
                if tmp_size == 0:
                    continue
                tmp2_size = tmp_size
                # 计算UAV从sensor收集到数据的标号
                for i, data in enumerate(sensor.data_buffer):
                    tmp2_size = tmp2_size - data[0]
                    if tmp2_size <= 0:
                        tmp_index = i
                        break
                tmp2_size = tmp2_size + sensor.data_buffer[tmp_index][0]
                # 加权平均age
                for i in range(tmp_index):
                    tmp_age = tmp_age + sensor.data_buffer[i][1] * (sensor.data_buffer[i][0] / tmp_size)
                tmp_age = tmp_age + sensor.data_buffer[tmp_index][1] * (tmp2_size / tmp_size)


                # UAV将从sensor中收集到的数据合并，存储到UAV缓冲区
                if sensor.no in agent.data_buffer.keys():
                    agent.data_buffer[sensor.no].append(tmp_size)
                else:
                    agent.data_buffer[sensor.no] = [tmp_size]
                # 每个sensor的数据属性(UAV收集sensor数据所需的时间)
                data_properties.append(tmp_size / sensor.trans_rate)
                # 在UAV的待计算数据中，加入收集到的数据（数据量，age（原始+收集时间-数据生成中加过的1），sensor编号）
                agent.total_data[sensor.no] = [tmp_size, tmp_age + (tmp_size / sensor.trans_rate) - 1, sensor.no]
                # agent.total_data[sensor.no] = [tmp_size, np.average([x[1] for x in sensor.data_buffer]), sensor.no]

                # 删除缓冲区中最老的XX%数据
                sensor.data_buffer[tmp_index][0] -= tmp2_size
                if sensor.data_buffer[tmp_index][0] == 0:
                    del sensor.data_buffer[tmp_index]
                for i in range(tmp_index):
                    del sensor.data_buffer[0]


        # UAV悬停时间
        # 若一个时隙内，UAV无法收集所有数据
        if data_properties:
            # UAV悬停时间 = 最长收集时间（在收集每个sensor中）
            hovering_time = max(data_properties)
            # print([data_properties, hovering_time])
            return hovering_time
        else:
            return 0
    # 完成收集
    # finish collection
    # 若无人机 正在收集
    elif not agent.idle:
        # UAV悬停时间（收集时间）-1
        hovering_time -= 1
        # 当收集空闲时
        if hovering_time <= 0:
            agent.idle = True
            # 对于所有UAV正在收集的sensor
            for no in agent.collecting_sensors:
                # 这些sensor更改为未被收集状态
                sensors[no].collect_state = False
            # UAV正在收集的sensor列表清空
            agent.collecting_sensors = []
            # 悬停时间设置为0
            hovering_time = 0
        return hovering_time
    else:
        return 0

# UAV卸载
def offloading(agent, center_pos, t=1):
    # 若UAV的计算完成数据为0
    if not agent.total_data:
        return (False, {})
    # age更新
    for k in agent.total_data.keys():
        agent.total_data[k][1] += t

    # UAV动作中需要进行卸载
    if sum(agent.action.offloading):
        # 提取待计算数据缓冲区（字典）中的所有 (key,value)
        data2process = [[k, d] for k, d in agent.total_data.items()]
        # 若 动作-计算 的索引超出了 待计算数据 的上限
        # 则将 动作-计算 的索引修正为 随机一个有效索引
        if agent.action.offloading.index(1) >= len(data2process):
            agent.action.offloading[agent.action.offloading.index(1)] = 0
            agent.action.offloading[np.random.randint(len(data2process))] = 1
            # print(self.action.execution)
        # UAV处于卸载忙碌状态
        agent.offloading_idle = False
        # UAV和中心基站的欧氏距离  # **
        dist = np.linalg.norm(np.array(agent.position)*5 - np.array(center_pos)*5)  # **
        dist = math.sqrt(dist*dist + 100*100)  # **
        # 卸载速率
        agent.trans_rate = trans_rate(dist, agent)  # to be completed
    else:
        return False, {}

    # 更新计算完成数据
    agent.total_data[data2process[agent.action.offloading.index(1)][0]][0] -= agent.trans_rate * t
    # 计算完成数据被全部卸载
    if agent.total_data[data2process[agent.action.offloading.index(1)][0]][0] <= 0:
        # 记录packet的传感器编号
        sensor_index = data2process[agent.action.offloading.index(1)][0]
        # 记录packet的AoI
        sensor_aoi = agent.total_data[data2process[agent.action.offloading.index(1)][0]][1]
        # 记录packet的数据量
        sensor_data = agent.data_buffer[sensor_index][0]
        # 在UAV缓冲区中删除这个packet
        del agent.data_buffer[sensor_index][0]
        # 在计算完成数据中删除这个
        del agent.total_data[data2process[agent.action.offloading.index(1)][0]]
        # UAV处于卸载空闲状态
        agent.offloading_idle = True
        # 返回完成标志和packet相关数据
        # return finish flag & total data
        return True, {sensor_index: [sensor_data, sensor_aoi]}
    return False, {}

# UAV-CC卸载速率  # ***
# 输入：dist距离, agent智能体
def trans_rate(dist, agent):  # to be completed
    W = 8e6 * agent.action.bandwidth  # 分配的带宽
    if W == 0:  # 若没分配到带宽
        return 0;
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / dist) - a)))  # LoS情况的概率
    fspl = (4 * np.pi * carrier_f * dist / (3e8))**2
    L = Pl * fspl * 10**(yita0 / 20) + 10**(yita1 / 20) * fspl * (1 - Pl)  # 平均A2G路径损耗
    rate = W * np.log2(1 + agent.ptr / (L * agent.noise * W))
    print('agent-{} rate: {},{},{},{},{}'.format(agent.no, dist, agent.action.bandwidth, Pl, L, rate))
    return rate


class MEC_world(object):
    def __init__(self, map_size, agent_num, sensor_num, obs_r, speed, collect_r, MAX_EP_STEPS, max_size=1, sensor_lam=0.5):
        # UAV列表
        self.agents = []
        # sensor列表
        self.sensors = []
        # 地图大小
        self.map_size = map_size
        # 地图中心坐标  # ***
        # self.center = (131, 42)  # ***
        # self.center = (random.randint(int(0.1 * map_size), int(0.9 * map_size)), random.randint(int(0.1 * map_size), int(0.9 * map_size)))  # 随机
        self.center = (map_size/2, map_size/2)  # 地图中心
        # sensor数量
        self.sensor_count = sensor_num
        # UAV数量
        self.agent_count = agent_num
        # 缓存区上限
        self.max_buffer_size = max_size
        # sensor带宽
        sensor_bandwidth = 1000
        # 生成数据大小上限
        max_ds = sensor_lam * 2
        # 数据生成速率
        data_gen_rate = 1
        # ？？？
        self.offloading_slice = 1
        self.execution_slice = 1
        self.time = 0
        # 数据源(data source, DS)地图
        self.DS_map = np.zeros([map_size, map_size])
        # 数据源(data source, DS)状态
        self.DS_state = np.ones([map_size, map_size, 2])
        # UAV悬停时间列表
        self.hovering_list = [0] * self.agent_count
        # UAV数据量列表
        self.tmp_size_list = [0] * self.agent_count
        # [self.tmp_size_list.append([0] * self.max_buffer_size) for i in range(self.agent_count)]
        # 卸载列表
        self.offloading_list = []
        # 计算结束的数据
        self.finished_data = []
        # UAV观察半径
        self.obs_r = obs_r
        # UAV移动速率
        self.move_r = speed
        # UAV覆盖范围
        self.collect_r = collect_r
        # sensor的age
        self.sensor_age = {}
        # random.seed(7)
        # 随机生成sensor的坐标
        self.sensor_pos = [random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num),
                           random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num)]
        # self.sensor_pos = [random.choices([i for i in range(int(0.1 * self.map_size), int(0.5 * self.map_size))], k=int(sensor_num / 2)) + random.choices(
        #     [i for i in range(int(0.5 * self.map_size), int(0.9 * self.map_size))], k=int(sensor_num / 2)), random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num)]
        # 对每个sensor
        for i in range(sensor_num):
            # 将sensor添加到world （此处包含了sensor所具有的关键属性）******************************
            self.sensors.append(
                Sensor(np.array([self.sensor_pos[0][i], self.sensor_pos[1][i]]), data_gen_rate, sensor_bandwidth, max_ds, self.max_buffer_size, lam=sensor_lam))
            self.sensor_age[i] = 0
            # 在地图上标记sensor的位置
            self.DS_map[self.sensor_pos[0][i], self.sensor_pos[1][i]] = 1
        # UAV的初始坐标
        self.agent_pos_init = [random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num),
                               random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num)]
        # 将UAV添加到world （此处包含了UAV所具有的关键属性）******************************
        for i in range(agent_num):
            self.agents.append(EdgeDevice(self.obs_r, np.array([self.agent_pos_init[0][i], self.agent_pos_init[1][i]]), speed, collect_r, self.max_buffer_size, MAX_EP_STEPS))

    def step(self):
        # 更新sensor的age列表（sensor的age和数据的age是独立的）
        for k in self.sensor_age.keys():
            self.sensor_age[k] += 1
        # age字典
        age_dict = {}
        # 数据生成 and 数据源状态更新
        logging.info("data generation")     # 日志输出提示信息——数据生成
        for sensor in self.sensors:
            # sensor生成数据(更新源数据缓冲区的age)
            sensor.data_gen()
            # sensor本地计算(仅更新本地计算数据包的age)
            s_finish_flag, s_data_dict = sensor.process()
            # 若本地计算完成，更新奖励状态
            if s_finish_flag:
                # sensor索引，[packet大小，packet的AoI]
                for sensor_id, data in s_data_dict.items():
                    # 将完成卸载的数据添加到计算结束数据
                    self.finished_data.append([data[0], data[1], sensor_id])
                    # 将packet的AoI添加到age字典中
                    if sensor_id in age_dict.keys():
                        age_dict[sensor_id].append([data[0], int(data[1])])
                    else:
                        age_dict[sensor_id] = [[data[0], int(data[1])]]

            # 若sensor缓冲区中有数据
            if sensor.data_buffer:
                # sensor缓冲区数据总量
                data_size = sum(i[0] for i in sensor.data_buffer)
                # 更新数据源状态，注意 (x,y) 反转为矩阵索引 (i,j)
                # [sensor缓冲区的数据总量，sensor缓冲区中数据的最大age]
                self.DS_state[sensor.position[1], sensor.position[0]] = [
                    data_size, sensor.data_buffer[0][1]]

        # 边缘计算 卸载 收集
        # edge process  offloading collect
        logging.info("edge operation")      # 日志输出提示信息——边缘操作
        # 对每个UAV进行循环
        for i, agent in enumerate(self.agents):
            # UAV边缘计算（更新UAV待计算数据缓冲区age）
            u_finish_flag, u_data_dict = agent.process()
            # 若UAV计算结束，更新奖励状态
            if u_finish_flag:
                # sensor索引，[数据大小，数据的AoI]
                for sensor_id, data in u_data_dict.items():
                    # 将完成卸载的数据添加到计算结束数据
                    self.finished_data.append([data[0], data[1], sensor_id])
                    # 将packet的AoI添加到age字典中
                    if sensor_id in age_dict.keys():
                        age_dict[sensor_id].append([data[0], int(data[1])])
                    else:
                        age_dict[sensor_id] = [[data[0], int(data[1])]]
                    # self.sensor_age[sensor_id] -=data[1]

            # UAV卸载到云（更新UAV计算完成数据缓冲区age）
            finish_flag, data_dict = offloading(agent, self.center)
            # 若卸载结束，更新奖励状态
            if finish_flag:
                # sensor索引，[packet大小，packet的AoI]
                for sensor_id, data in data_dict.items():
                    # 将完成卸载的数据添加到计算结束数据
                    self.finished_data.append([data[0], data[1], sensor_id])
                    # 将packet的AoI添加到age字典中
                    if sensor_id in age_dict.keys():
                        age_dict[sensor_id].append([data[0], int(data[1])])
                    else:
                        age_dict[sensor_id] = [[data[0], int(data[1])]]
                    # self.sensor_age[sensor_id] -=data[1]
            # 收集
            self.hovering_list[i] = data_collecting(self.sensors, agent, self.hovering_list[i])
            # print(self.hovering_list[i])
        # 对每个sensor的标号进行循环
        for id in age_dict.keys():
            # 对某个sensor的packet的age字典进行排序（升序）
            # 将sensor最小的packet的age作为sensor的age
            # self.sensor_age[id] = sorted(age_dict[id])[0]
            # 将sensor所有计算结束数据的age，求加权平均作为sensor的age
            tmp_size = 0  # sensor在本时隙计算结束的数据量的总和
            tmp_age = 0
            for age_data in age_dict[id]:
                tmp_size += age_data[0]
                tmp_age += age_data[0] * age_data[1]
            self.sensor_age[id] = int(tmp_age / tmp_size)
        # 打印UAV的等待时间列表
        print('hovering:{}'.format(self.hovering_list))