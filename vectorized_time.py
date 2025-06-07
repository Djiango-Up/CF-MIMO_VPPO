from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
import numpy as np
from environment import highway_cf
from pathlib import Path
import scipy.io as scio

L = 15
K = 300
Cap = 6
dis = 10
eps = 1.1
base_alpha = 0.8

# 初始化文件数F的起始值
initial_F = 10

def make_env(rank, files, seed=0):
    def _init():
        env = highway_cf(AP=L, UE=K, Cap=Cap, dis=dis, files=files, epsilon=eps, alpha=base_alpha)
        env.seed(seed + rank)
        return env
    return _init


if __name__ == '__main__':
    n_envs = 3  # 并行环境的数量
    times = np.zeros(50)  # 存储每次迭代的平均时间

    for f in range(50):
        F = 10 + f
        envs = SubprocVecEnv([make_env(i, F) for i in range(n_envs)])
        # model = SAC('MlpPolicy', env=envs, learning_rate=5e-4, batch_size=128, policy_kwargs={"net_arch": [256, 256]},
        #             verbose=1,tensorboard_log="./tensorboard/SAC/")
        model = PPO('MlpPolicy',
                    env=envs,
                    learning_rate=3e-4,  # 原参数是3e-4 降低该学习率看看奖励能不能稳定上升
                    batch_size=128,
                    policy_kwargs=dict(net_arch=[256, 256]),  # 可以尝试提高复杂度512*512 #网络架构
                    n_steps=2048,  # 每个epoch的采样步数
                    gamma=0.99,  # 折扣因子
                    # GAE方法 https://zhuanlan.zhihu.com/p/675921649
                    gae_lambda=0.95,  # 广义优势估计的衰减参数  该参数越小 越关注即时奖励
                    clip_range=0.1,  # 裁剪比例
                    n_epochs=10,  # 每个批次的优化迭代次数
                    ent_coef=0.001,  # 熵奖励权重 如果偏向命中率的话 降低折扣因子；反之考虑长期奖励，可以提高
                    verbose=1,  # 日志信息输出
                    tensorboard_log="./tensorboard/PPO/")
        model.learn(total_timesteps=40000)

        total_interaction_time = 0  # 存储所有环境交互的总时间
        for _ in range(1000):  # 假设进行1000次环境交互
            obs = envs.reset()

            start_time = time.time()
            actions, _states = model.predict(obs)
            # obs, rewards, dones, info = envs.step(actions)
            end_time = time.time()

            total_interaction_time += (end_time - start_time)

        #在所有并行环境实例中执行一次动作预测所需的平均时间
        times[f] = total_interaction_time /  1000  /3 # 计算平均每个环境与算法的交互时间

    # 保存时间数据
    a = Path.cwd()
    scio.savemat(a / "MAT_datas/PPO_vectorized_time_3.mat", {"time_PPO_vectorized": times})