from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
import scipy.io as scio
from environment import highway_cf
import pandas as pd
import torch
import numpy as np
from torch.optim import Adam
from torch import nn
from torch.distributions import Normal
from stable_baselines3 import TD3
from stable_baselines3 import SAC

'''
通过使用Vectorized异步训练 ,并行和多个环境交互，提高样本效率和训练速度
'''

L = 15
K = 300
Cap = 6
F = 20
dis = 10
eps = 1.1
base_alpha=0.8
#自定义环境 AP是接入口 UE用户设备 cap缓存容量 f文件数量 eps Zipf分布的偏度参数
env = highway_cf(AP=L, UE=K, Cap=Cap, dis=dis, files=F, epsilon=eps,alpha=base_alpha)

def make_env(rank, env_type):
    def init_env():
        # 初始化环境，传入rank以区分不同的环境实例
        env = highway_cf(AP=L, UE=K, Cap=Cap, dis=dis, files=F, epsilon=eps, alpha=base_alpha)
        return env
    return init_env



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # 创建向量化环境
    n_envs = 3  # 并行环境的数量
    env_type = 'highway_cf'  # 环境类型名称
    envs = SubprocVecEnv([make_env(i, env_type) for i in range(n_envs)])

    model = PPO('MlpPolicy',  # 'MlpPolicy'
                env=envs,
                learning_rate=3e-4, #3e-4
                batch_size=256,  # 原：128
                policy_kwargs=dict(net_arch=[256,256]),  # 256*256
                n_steps=2048,  # 每个epoch的采样步数 2048
                gamma=0.99,  # 折扣因子
                gae_lambda=0.95,  # 广义优势估计的衰减参数
                clip_range=0.1,  # 裁剪比例,原：0.1
                n_epochs=10,  # 每个批次的优化迭代次数
                ent_coef=0.001,  # 熵奖励权重 0.001
                verbose=1,
                tensorboard_log="./tensorboard/PPO/")

    model.learn(total_timesteps=200000)
    model.save("PPO")
    del model
    model = PPO.load("PPO", env=envs)

    model = SAC('MlpPolicy',
                env=env,
                learning_rate=5e-4,
                batch_size=128,
                policy_kwargs={"net_arch": [256, 256]},
                verbose=1,
                tensorboard_log="./tensorboard/SAC/"
                ).learn(total_timesteps=200000)

    model.save("SAC")
    del model
    model = SAC.load("SAC", env=envs)


    # model = TD3('MlpPolicy',     #'MlpPolicy'
    #             env=envs,
    #             learning_rate=3e-4,
    #             buffer_size=10000,  # 重播缓冲区大小，存储经验 10000
    #             learning_starts=1000, # 开始学习之前的环境步数，只探索环境不学习 1000
    #             batch_size=128,       # 每个训练步骤的样本数量 128
    #             tau=0.005,            # 软更新的目标网络系数，将目标网络更新为主网络 0.005
    #             gamma=0.99,           # 折扣因子 用于计算折扣奖励 0.99
    #             train_freq=1000,      # 训练频率，每个多少步模型更新一次 1000
    #             gradient_steps=1000,  # 每次更新模型的优化步数 1000
    #             action_noise=None,    # 动作噪声，用于探索  None
    #             policy_delay=2,       # 延迟更新策略和目标网络的频率**  2
    #             target_policy_noise=0.2,    #这是目标策略噪声，用于在计算目标Q值时对目标动作添加噪声**  0.2
    #             target_noise_clip=0.5,      #目标策略噪声的裁剪范围，用于计算目标Q值时对目标动作添加噪声。 0.5
    #             verbose=1,
    #             tensorboard_log="./tensorboard/TD3/")
    # model.learn(total_timesteps=40000)   # 20000
    # model.save("TD3")
    # del model
    # model = TD3.load("TD3", env)

    # model = A2C('MlpPolicy', envs, learning_rate=3e-4, n_steps=2048)
    # model.learn(total_timesteps=40000)
    # model.save("A2C")
    # del model
    # model = A2C.load("A2C", envs)
    
    obs = envs.reset()

    rewards_per_step = []
    hits_per_step = []
    delays_per_step = []

    for i in range(10000): #10000
        action, _states = model.predict(obs)
        obs, rewards, done, info = envs.step(action)
        step_delay = np.mean([info_env.get('delay', 0) for info_env in info])

        rewards_per_step.append(np.mean(rewards))
        hits_per_step.append(np.mean(envs.get_attr('hit')))
        delays_per_step.append(step_delay)

    # # 将列表转换为 numpy 数组
    # rewards_per_step = np.array(rewards_per_step)
    # hits_per_step = np.array(hits_per_step)
    # delays_per_step = np.array(delays_per_step)
    #
    # # 创建 DataFrame 并保存为 CSV 文件
    # df_rewards = pd.DataFrame(rewards_per_step.reshape(-1, 1))
    # df_rewards.to_csv("./MAT_datas/Vectorized_A2C_rewards.csv", index=False, header=False)
    #
    # df_hits = pd.DataFrame(hits_per_step.reshape(-1, 1))
    # df_hits.to_csv("./MAT_datas/Vectorized_A2C_hits.csv", index=False, header=False)
    #
    # df_delays = pd.DataFrame(delays_per_step.reshape(-1, 1))
    # df_delays.to_csv("./MAT_datas/Vectorized_A2C_delays.csv", index=False, header=False)
    a = Path.cwd()

    scio.savemat(a / "MAT_datas/Vectorized_SAC_QoE（4）.mat", {"Vectorized_SAC_QoE": rewards_per_step})
    scio.savemat(a / "MAT_datas/Vectorized_SAC_hit（4）.mat", {"Vectorized_SAC_hit": hits_per_step})
    scio.savemat(a / "MAT_datas/Vectorized_SAC_delay（4）.mat", {"Vectorized_SAC_delay": delays_per_step})
