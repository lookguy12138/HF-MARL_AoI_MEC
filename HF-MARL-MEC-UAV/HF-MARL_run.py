import gym
import numpy as np
import random
from MEC_env import mec_def
from MEC_env import mec_env
import tensorflow as tf
from tensorflow import keras
import tensorboard
import datetime
import MAAC_agent
from matplotlib import pyplot as plt
import json
import os

# tensorflow 中解决显存不足的问题
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True  # 程序按需申请内存1e-13
tf.compat.v1.Session(config=config)

print("TensorFlow version: ", tf.__version__)  # 打印TF版本
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # 打印可用GPU数量
plt.rcParams['figure.figsize'] = (9, 9)  # 设置图像显示大小为(9, 9)
# logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# 设置参数
map_size = 200  # 地图大小 200*10
agent_num = 4  # agent（UAV）数量
sensor_num = 30  # sensor数量
obs_r = 60  # 观察半径 60*10
collect_r = 40  # 收集半径 ** 40*40
speed = 6  # UAV移动速率 6*10
max_size = 10  # UAV缓冲区上限（带计算数据缓冲区，计算完成数据缓冲区）
sensor_lam = 1e3 + 10 * 200  # sensor数据生成速率（到达率）：①poisson分布中，间隔时间的期望；②lam*2是sensor生成数据大小的上限 1e3 + 0 * 200
# i = 0, 2, 4， 6， 8,10

MAX_EPOCH = 5000
MAX_EP_STEPS = 200
LR_A = 0.001  # actor的学习率 learning rate for actor
LR_C = 0.002  # critic的学习率 learning rate for critic
GAMMA = 0.85  # reward折扣（系统惩罚的衰变系数） reward discount
TAU = 0.8  # 软替换（# target网络更新中，保留原始target网络参数的比例系数） soft replacement
BATCH_SIZE = 64
alpha = 0.9  # alpha参数（画图相关参数）
beta = 0.1  # beta参数（没用到）
Epsilon = 0.2  # UAV随机探索概率

# 固定随机种子以重现结果 random seeds are fixed to reproduce the results
map_seed = 1
rand_seed = 17
up_freq = 8  # 联邦更新周期，同时也是target网络更新周期
render_freq = 32  # 画图（MEC环境示意图）间隔
FL = True
FL_omega = 0.5  # 联邦因子
np.random.seed(map_seed)  # 生成指定的随机数
random.seed(map_seed)  # 生成指定的随机数
tf.random.set_seed(rand_seed)  # 生成指定的随机数

# 根据上面参数得到的参数字典
params = {
    'map_size': map_size,
    'agent_num': agent_num,
    'sensor_num': sensor_num,
    'obs_r': obs_r,
    'collect_r': collect_r,
    'speed': speed,
    'max_size': max_size,
    'sensor_lam': sensor_lam,

    'MAX_EPOCH': MAX_EPOCH,
    'MAX_EP_STEPS': MAX_EP_STEPS,
    'LR_A': LR_A,
    'LR_C': LR_C,
    'GAMMA': GAMMA,
    'TAU': TAU,
    'BATCH_SIZE': BATCH_SIZE,
    # 'alpha': alpha,
    # 'beta': beta,
    'Epsilon': Epsilon,
    'learning_seed': rand_seed,
    'env_seed': map_seed,
    'up_freq': up_freq,
    'render_freq': render_freq,
    'FL': FL,
    'FL_omega': FL_omega
}
# 生成MEC world
# 输入：地图大小，agent数量，sensor数量，观察半径，速度，收集半径，缓冲区上限，sensor的lam参数
mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, MAX_EP_STEPS, max_size,
                              sensor_lam)
# 生成MEC环境
# 输入：MECworld，alpha参数，beta参数
env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta)

#
MAAC = MAAC_agent.MAACAgent(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)

# 格式化显示当前日期时间的函数 （这行代码的时间显示格式为：年月日-时分秒）
m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
# 将整个程序的参数字典（python对象）转换为适当的json对象，并存储到相应位置
f = open('logs/hyperparam/%s.json' % m_time, 'w')
json.dump(params, f)
f.close()
#
MAAC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq, FL=FL, FL_omega=FL_omega)

