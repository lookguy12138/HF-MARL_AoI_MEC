# -*- coding: UTF-8 -*-
import numpy as np
import gym
from gym import spaces
import numpy as np
from .space_def import circle_space
from .space_def import onehot_space
from .space_def import sum_space
from gym.envs.registration import EnvSpec
import logging
from matplotlib import pyplot as plt
from IPython import display

# 设置日志记录器的显示级别
logging.basicConfig(level=logging.WARNING)

# plt.figure()
# plt.ion()


# 获得圆形图
# 输入：（UAV坐标，UAV的观察/收集半径）
# 输出：和UAV距离为r的点的坐标
def get_circle_plot(pos, r):
    x_c = np.arange(-r, r, 0.01)
    up_y = np.sqrt(r**2 - np.square(x_c))
    down_y = - up_y
    x = x_c + pos[0]
    y1 = up_y + pos[1]
    y2 = down_y + pos[1]
    return [x, y1, y2]


class MEC_MARL_ENV(gym.Env):
    # 元数据？？？
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, world, alpha=0.5, beta=0.2, aggregate_reward=False, discrete=True,
                 reset_callback=None, info_callback=None, done_callback=None):
        # 系统初始化
        self.world = world  # MEC_world
        self.obs_r = world.obs_r  # UAV观察半径
        self.move_r = world.move_r  # UAV移动速率
        self.collect_r = world.collect_r  # UAV覆盖半径
        self.max_buffer_size = self.world.max_buffer_size  # 缓冲区上限
        self.agents = self.world.agents  # UAV agent 列表
        self.agent_num = self.world.agent_count  # UAV数量
        self.sensor_num = self.world.sensor_count  # sensor数量
        self.sensors = self.world.sensors  # sensor 列表
        self.DS_map = self.world.DS_map  # 数据源地图
        self.map_size = self.world.map_size  # 地图尺寸
        self.DS_state = self.world.DS_state  # 数据源状态
        self.alpha = alpha
        self.beta = beta

        # bool型参数 ？？？
        self.reset_callback = reset_callback  # ？？？
        self.info_callback = info_callback  # ？？？
        self.done_callback = done_callback  # ？？？

        # 游戏模式 ？？？ (game mode)
        self.aggregate_reward = aggregate_reward  # 分享相同的rewards（bool） (share same rewards)
        self.discrete_flag = discrete  # 离散flag（bool）
        self.state = None  # UAV的观察
        self.time = 0  # ？？？
        self.images = []  # ？？？

        # 构形空间 (configure spaces)
        self.action_space = []  # UAV动作空间列表
        self.observation_space = []  # UAV观察空间列表

        # reward中公平指数所需的工具参数
        self.f_up = 0
        self.f_down = [0, 0, 0, 0]

        # 对所有的UAV
        for agent in self.agents:
            # 若离散
            if self.discrete_flag:
                # 利用spaces.Tuple()创建动作空间 （UAV移动速率，卸载，带宽/UAV数量？？？，执行）
                act_space = spaces.Tuple((circle_space.Discrete_Circle(
                    agent.move_r), onehot_space.OneHot(self.max_buffer_size), sum_space.SumOne(self.agent_num), onehot_space.OneHot(self.max_buffer_size)))
                # move, offloading(boolxn), bandwidth([0,1]), execution
                # 利用space.Tuple()创建观察空间（位置，观察地图）
                obs_space = spaces.Tuple((spaces.MultiDiscrete(
                    [self.map_size, self.map_size]), spaces.Box(0, np.inf, [agent.obs_r * 2, agent.obs_r * 2, 2])))  # 位置：多离散动作空间  观察地图：下限0，上限无穷的连续空间
                # pos, obs map
                self.action_space.append(act_space)
                self.observation_space.append(obs_space)
        self.render()

    # environment update
    def step(self, agent_action, center_action, sensor_action):
        obs = []
        reward = []
        average_age = []
        fairness_index = []
        done = []
        info = {'n': []}
        self.agents = self.world.agents
        self.sensors = self.world.sensors

        # world step
        logging.info("set actions")  # 日志输出
        # 对每个sensor循环
        for i, sensor in enumerate(self.sensors):
            # 设置sensor的动作
            self._set_sensor_action(sensor_action[i], sensor)
        # 对每个UAV循环
        for i, agent in enumerate(self.agents):
            # 设置UAV的动作
            self._set_action(agent_action[i], center_action, agent)

        # world update
        self.world.step()
        # 新的观察 (new observation)
        logging.info("agent observation")
        for agent in self.agents:
            obs.append(self.get_obs(agent))
            done.append(self._get_done(agent))
            # reward.append(self._get_age() / self._get_fairness_index())  # 所有UAV的reward是相同的
            reward.append(self._get_age())  # 所有UAV的reward是相同的
            average_age.append(self._get_age())
            fairness_index.append(self._get_fairness_index())
            info['n'].append(self._get_info(agent))
        self.state = obs
        # reward （这部分没有运行）
        reward_sum = np.sum(reward)
        logging.info("get reward")
        if self.aggregate_reward:
            reward = [reward_sum] * self.agent_num
        return self.state, reward, average_age, fairness_index, done, info

    # 重置世界以及世界中的sensor和UAV
    def reset(self):
        # 重置world (reset world)
        self.world.finished_data = []
        # reset renderer
        # self._reset_render()
        # record observations for each agent???
        # 对每个sensor
        for sensor in self.sensors:
            sensor.data_buffer = []
            sensor.collect_state = False

            sensor.finish_data = 0
            sensor.total_data = [0, 0]
            sensor.computing_state = False
        # 对每个UAV
        for agent in self.agents:
            agent.idle = True
            agent.data_buffer = {}
            agent.total_data = {}
            agent.collecting_sensors = []

    # 设置动作
    def _set_action(self, act, center_action, agent):
        agent.action.move = np.zeros(2)  # UAV移动
        agent.action.execution = act[1]  # 设置UAV执行
        agent.action.bandwidth = center_action[agent.no]  # 卸载带宽
        # 若①UAV可移动；②UAV收集空闲
        if agent.movable and agent.idle:
            # print([agent.no, act[0]])
            # 若UAV动作的移动距离 > UAV移动速率
            if np.linalg.norm(act[0]) > agent.move_r:  # act[0]：(x轴移动距离, y轴移动距离)
                # 修正：令UAV动作的移动距离 = UAV移动速率
                act[0] = [int(act[0][0] * agent.move_r / np.linalg.norm(act[0])), int(act[0][1] * agent.move_r / np.linalg.norm(act[0]))]
            # 若①UAV动作的移动距离为0；②50%的情况下
            if not np.count_nonzero(act[0]) and np.random.rand() > 0.5:
                # 正态分布生成UAV动作的移动距离
                mod_x = np.random.normal(loc=0, scale=1)
                mod_y = np.random.normal(loc=0, scale=1)
                mod_x = int(min(max(-1, mod_x), 1) * agent.move_r / 2)
                mod_y = int(min(max(-1, mod_y), 1) * agent.move_r / 2)
                act[0] = [mod_x, mod_y]
            # 设置UAV移动
            agent.action.move = np.array(act[0])  # np.array()将列表转化为有维度的数组
            # UAV移动的边界处理
            new_x = agent.position[0] + agent.action.move[0]
            new_y = agent.position[1] + agent.action.move[1]
            if new_x < 0 or new_x > self.map_size - 1:
                agent.action.move[0] = -agent.action.move[0]
            if new_y < 0 or new_y > self.map_size - 1:
                agent.action.move[1] = -agent.action.move[1]
            # UAV位置坐标更新
            agent.position += agent.action.move
            # agent.position = np.array([max(0, agent.position[0]),
            #                            max(0, agent.position[1])])
            # agent.position = np.array([min(self.map_size - 1, agent.position[0]), min(
            #     self.map_size - 1, agent.position[1])])
        # 若UAV卸载空闲
        if agent.offloading_idle:
            # 设置UAV卸载
            agent.action.offloading = act[2]
        # 动作：①act[0]-移动；②act[1]-执行；③act[2]-卸载；④center_action[agent.no]-卸载带宽
        print('agent-{} action: move{}, exe{},off{},band{}'.format(agent.no, agent.action.move, agent.action.execution, agent.action.offloading, agent.action.bandwidth))

    # 设置sensor的动作
    def _set_sensor_action(self, act, sensor):
        # 若sensor卸载(UAV收集)空闲
        if not sensor.computing_state:
            sensor.action.computing = act[0]
        # 动作：①act[0]-本地计算；②act[1]-被收集
        print('sensor-{} action: computing{} collecting{}'.format(sensor.no, sensor.action.computing[0], 1))

    # 获取用于基准测试的信息 (get info used for benchmarking)
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # 获得某个agent的观察--观察区域内的数据源状态[sensor的数据总量，sensor中数据的最大age] (get observation for a particular agent)
    def get_obs(self, agent):
        obs = np.zeros([agent.obs_r * 2 + 1, agent.obs_r * 2 + 1, 2])
        # UAV观察范围的左上点和右下点
        # 左上点 (left up point)
        lu = [max(0, agent.position[0] - agent.obs_r),
              min(self.map_size, agent.position[1] + agent.obs_r + 1)]
        # 右下点 (right down point)
        rd = [min(self.map_size, agent.position[0] + agent.obs_r + 1),
              max(0, agent.position[1] - agent.obs_r)]

        # 观察地图的坐标 (ob_map position)
        ob_lu = [agent.obs_r - agent.position[0] + lu[0],
                 agent.obs_r - agent.position[1] + lu[1]]
        ob_rd = [agent.obs_r + rd[0] - agent.position[0],
                 agent.obs_r + rd[1] - agent.position[1]]
        # print([lu, rd, ob_lu, ob_rd])
        for i in range(ob_rd[1], ob_lu[1]):
            map_i = rd[1] + i - ob_rd[1]
            # print([i, map_i])
            obs[i][ob_lu[0]:ob_rd[0]] = self.DS_state[map_i][lu[0]:rd[0]]
        # print(self.DS_state[ob_rd[1]][ob_lu[0]:ob_rd[0]].shape)
        agent.obs = obs
        # print(obs.shape)
        return obs

    # 获取状态地图
    def get_statemap(self):
        sensor_map = np.ones([self.map_size, self.map_size, 2])
        agent_map = np.ones([self.map_size, self.map_size, 2])
        # sensor状态地图
        for sensor in self.sensors:
            # sensor缓冲区中的数据量总和
            # sensor缓冲区中的数据平均年龄
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][0] = sum([i[0] for i in sensor.data_buffer])
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][1] = sum([i[1] for i in sensor.data_buffer]) / max(len(sensor.data_buffer), 1)
        # UAV状态地图
        for agent in self.agents:
            # UAV计算完成数据的数据量总和
            # UAV计算完成数据的的数据平均年龄
            agent_map[int(agent.position[1])][int(agent.position[0])][0] = sum([i[0] for i in list(agent.total_data.keys())])
            agent_map[int(agent.position[1])][int(agent.position[0])][1] = sum([i[1] for i in list(agent.total_data.keys())]) / max(len(list(agent.total_data.keys())), 1)
        return sensor_map, agent_map
        # get dones for a particular agent
        # unused right now -- agents are allowed to go beyond the viewing screen

    # 获得云中心状态
    def get_center_state(self):
        # 所有UAV计算完成数据列表
        buffer_list = np.zeros([self.agent_num, 2, self.max_buffer_size])
        # UAV位置坐标列表
        pos_list = np.zeros([self.agent_num, 2])
        # 对所有UAV循环
        for i, agent in enumerate(self.agents):
            # 获得每个UAv的位置
            pos_list[i] = agent.position
            # 获得每个UAV计算完成的数据
            for j, k in enumerate(list(agent.total_data.keys())):
                buffer_list[i][0][j] = agent.total_data[k][0]
                buffer_list[i][1][j] = agent.total_data[k][1]
        # print(buffer_list)
        # print(pos_list)
        return buffer_list, pos_list

    # 获得缓冲区状态
    def get_buffer_state(self):
        # 各个UAV
        exe = []
        done = []
        # 每个UAV待计算和计算完成的包的个数
        for agent in self.agents:
            exe.append(len(agent.total_data))
        return exe, done

    # 获得done ？？？
    def _get_done(self, agent):
        if self.done_callback is None:
            return 0
        return self.done_callback(agent, self.world)  # ？？？

    # 获得平均age (average age)
    def _get_age(self):
        return np.mean(list(self.world.sensor_age.values()))

    # 获得UAV移动公平指数
    def _get_fairness_index(self):
        r_t = 0
        c_m_t = []
        tmp = [0, 0, 0, 0]
        # 计算c_m_t
        for i, agent in enumerate(self.agents):
            r_t += agent.t_distance
        if r_t == 0:  # 若四个无人机都没动
            return 1;
        for i, agent in enumerate(self.agents):
            c_m_t.append(agent.t_distance / r_t)
        # 计算f_t
        for i, agent in enumerate(self.agents):
            self.f_up += agent.t_distance
        for i, agent in enumerate(self.agents):
            self.f_down[i] += agent.t_distance
        up = self.f_up ** 2
        for i, value in enumerate(self.f_down):
            tmp[i] = value ** 2
        down = sum(tmp) * len(self.agents)
        f_t = up / down
        return f_t

    # 获得某个特定agent的奖励 (get reward for a particular agent)
    # 输出：sensor的平均age（所有UAV的reward都一样）
    def _get_reward(self):
        return np.mean(list(self.world.sensor_age.values()))
        # state_reward = sum(sum(self.DS_state)) / self.sensor_num
        # done_reward = [[i[0], i[1]] for i in self.world.finished_data]
        # if not done_reward:
        #     done_reward = np.array([0, 0])
        # else:
        #     # print(np.array(done_reward))
        #     done_reward = np.average(np.array(done_reward), axis=0)
        # buffer_reward = 0
        # for agent in self.agents:
        #     if agent.done_data:
        #         buffer_reward += np.mean([d[1] for d in agent.done_data])
        # buffer_reward = buffer_reward / self.agent_num
        # # print(buffer_reward)
        # # print([state_reward, done_reward])
        # return self.alpha * done_reward[1] + self.beta * (state_reward[1] + self.sensor_num - self.map_size * self.map_size) + (1 - self.alpha - self.beta) * buffer_reward

    # 画图
    def render(self, name=None, epoch=None, save=False):
        # plt.subplot(1,3,1)
        # plt.scatter(self.world.sensor_pos[0],self.world.sensor_pos[1],alpha=0.7)
        # plt.grid()
        # plt.title('sensor position')
        # plt.subplot(1,3,2)
        # plt.scatter(self.world.agent_pos_init[0],self.world.agent_pos_init[1],alpha=0.7)
        # plt.grid()
        # plt.title('agent initial position')
        # plt.subplot(1,3,3)
        plt.figure()
        # 画出所有sensor
        plt.scatter(self.world.sensor_pos[0], self.world.sensor_pos[1], c='cornflowerblue', alpha=0.9)  # 画散点图

        # 对每个UAV
        for agent in self.world.agents:
            # 画出UAV
            plt.scatter(agent.position[0], agent.position[1], c='orangered', alpha=0.9)
            # 标注文字（UAV标号）
            plt.annotate(agent.no + 1, xy=(agent.position[0], agent.position[1]), xytext=(agent.position[0] + 0.1, agent.position[1] + 0.1))
            # 获得UAV可以观察到的边界
            obs_plot = get_circle_plot(agent.position, self.obs_r)
            # 获得UAV可以收集到的边界
            collect_plot = get_circle_plot(agent.position, self.collect_r)
            # 用颜色填充两条曲线之间的区域(x, y1, y2) -- 观察obs
            plt.fill_between(obs_plot[0], obs_plot[1], obs_plot[2], where=obs_plot[1] > obs_plot[2], color='darkorange', alpha=0.02)
            # 用颜色填充两条曲线之间的区域(x, y1, y2) -- 收集collect
            plt.fill_between(collect_plot[0], collect_plot[1], collect_plot[2], where=collect_plot[1] > collect_plot[2], color='darkorange', alpha=0.05)
        plt.grid()  # 添加网格
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Sensors', 'Edge Agents'])  # 添加注释
        plt.axis('square')  # 作图为正方形，并且x,y轴范围相同
        plt.xlim([0, self.map_size])  # x轴的作图范围
        plt.ylim([0, self.map_size])  # y轴的作图范围
        plt.title('all entity position(epoch%s)' % epoch)
        # 若不保存图片，则展示图片
        if not save:
            plt.show()
            return
        # 保存图片
        plt.savefig('%s/%s.png' % (name, epoch))
        plt.close()
        # plt.pause(0.3)
        # plt.show()

    def close(self):
        return None
