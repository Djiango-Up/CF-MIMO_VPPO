from abc import ABC
import gym
from gym import spaces
import numpy as np
import torch


def tau_QoE(cache, req, cap, file, AP, file_size, dis, weight, need, UE,alpha):
    SE = 200                                
    satisfy = np.zeros((UE, 1))

    T = np.array([1, 20])
    taus = np.zeros((AP, file))             
    #np.sum(cache, axis=0).reshape((file, 1) 
    # print(file_size)
    # print(file.shape())
    a = np.maximum(file_size - np.sum(cache, axis=0).reshape((file, 1)) * cap, np.zeros_like(file_size))
    for i in range(file):                   
        taus[:, i] = -a[i] * T[1]          
        for j in range(AP):
            distance = dis * abs(np.arange(AP) - j)         
            for num in range(AP):                           
                b = int(min(AP - 1, j + num))              
                c = int(max(0, j - num))                    
                if cache[c:b + 1, i].sum() * cap + a[i] >= file_size[i]:    
                    taus[j, i] = taus[j, i] - np.max(cache[c:b + 1, i] * distance[c:b + 1]) * cap * T[0]  
                    break

    single = np.zeros((file, 1))             
    file_delay = np.zeros((file, 1))
    
    for i in range(file):
        single[i] = np.min(taus[:, i])

    t = 0
    QoE = 0
    middle = -(weight + req).reshape(-1)   #文件权重和请求次数向量相加
    line = np.argsort(middle)              #按照middle数组升序，可以率先考虑请求次数更多或者权重更大的文件
    delay = 0                              #自定义奖励函数的延迟权重因子
    for i in line:                         #迭代处理每个文件的请求 更新QoE
        t = t + single[i]
        #1/SE 每个请求单位对延迟的敏感度 SE越大表示系统对延迟的敏感度较低
        #后续在调整QoE加入延迟考虑 应该调整这个SE 使得延迟敏感提高
        t = t - req[i] * (1 / SE)

        if t > -500:
            QoE = QoE + req[i] + weight[i]                 
            satisfy[np.where(need == i)] = 1               
            weight[i] = 0
            # delay = delay + single[i] * (req[i] + weight[i])  
            delay = delay + single[i]
        else:
            weight[i] = weight[i] + req[i]                 
            delay_pro = weight[i] + req[i] + file_size[i]
            delay = delay + delay_pro

    # return np.array(QoE/((weight + req).sum())), satisfy, weight #源码
    total_delay = np.sum(delay)
    return np.array( alpha * QoE / ((weight + req).sum()) + (alpha-1)*delay/(req+weight+file_size+single).sum() ), satisfy, weight, total_delay

#hit2用在了环境，奖励函数中
def hit_2(cache, F, cap, file_size, req, K):
    r = 0
    a = np.minimum(np.sum(cache, axis=0).reshape((F, 1)) * cap / file_size, np.ones_like(file_size))
    for f in range(F):
        r = r + req[f] * a[f]   #a[]是该文件的缓存比例
    return r / K


def hit_pro(cache, F, cap, file_size, req, K):          
    r = 0
    for f in range(F):
        if cache[:, f].sum() * cap >= file_size[f]:    
            r = r + req[f]                              
    return r - K


class highway_cf(gym.Env, ABC):
    def __init__(self, AP, UE, Cap, files, dis, epsilon,alpha):
        self.cap = Cap

        self.L = AP
        self.K = UE
        self.F = files
        self.Cap = Cap
        self.dis = dis
        self.epsilon = epsilon
        #这里能看到输入维度由环境的状态空间决定 ，状态空间是一个(35,)的向量（L + 1 = 15 + 1和F = 20，所以(15 + 1) * 20 = 35）
        #输出维度由动作空间决定 动作空间是一个(300,)的向量（L * F = 15 * 20）。
        self.observation_space = spaces.Box(0, 1, ((self.L + 1) * self.F,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (self.L * self.F,), dtype=np.float32)

        self.req = np.zeros((1, self.F))
        self.need = np.zeros(self.K)

        self.state = None
        self.file = np.array(range(self.F))
        self.file_size = np.random.randint(5, 10, self.F)
        self.file_size = self.file_size.reshape((self.F, 1))
        self.zipf = np.zeros(self.F)

        self.patience_value = np.random.randint(2, 5, (self.K, 1))
        self.patience = self.patience_value
        self.satisfy = np.ones((self.K, 1))
        self.weight = np.zeros((self.F, 1))

        self.hit = 0
        self.QoE = 0

        #请求参数用来动态调整延迟因子 tau_QoE 中 alpha
        self.total_requests = 0
        self.successful_requests = 0
        self.alpha = alpha

    def Zipf(self):
        for j in range(self.F):
            self.zipf[j] = ((self.file[j] + 1) ** (-self.epsilon))
        a = self.zipf.sum()
        self.zipf = self.zipf / a      
    def request(self):

        #更新请求计数器，用于用来动态调整延迟因子 tau_QoE 中 alpha
        self.total_requests+=self.K
        self.successful_requests += np.sum(self.satisfy)

        self.req = np.zeros((self.F, 1))
        for i in range(self.K):
            if self.satisfy[i] == 1:
                self.patience[i] = self.patience_value[i]
                req = np.random.choice(a=self.file, size=1, replace=True, p=self.zipf)
                self.need[i] = req
                self.req[req] = self.req[req] + 1
            else:
                if self.patience[i] > 0:
                    self.req[int(self.need[i])] = self.req[int(self.need[i])] + 1
                    self.patience = self.patience - 1
                else:
                    self.patience[i] = self.patience_value[i]
                    req = np.random.choice(a=self.file, size=1, replace=True, p=self.zipf)
                    self.weight[int(self.need[i])] = np.max(self.weight[int(self.need[i])]-1, 0)
                    self.need[i] = req
                    self.req[req] = self.req[req] + 1


    def reward(self, actions):
        cache = actions.reshape((self.L, self.F))
        cache = np.array(torch.softmax(torch.from_numpy(cache), 1))
        alpha = self.adjust_alpha_based_on_success_rate()

        r_hit = hit_2(cache=cache, F=self.F, cap=self.Cap, file_size=self.file_size, req=self.req, K=self.K)
        r_QoE, self.satisfy, self.weight,total_delay = \
            tau_QoE(cache=cache, req=self.req, cap=self.Cap, file_size=self.file_size, file=self.F, dis=self.dis,
                    AP=self.L, need=self.need, weight=self.weight, UE=self.K,alpha=alpha)
        self.hit = r_hit
        self.QoE = r_QoE
        return r_QoE, total_delay
        # return r_QoE
    #奖励函数
    def step(self, actions):
        self.request()
        self.state = self.state.reshape((self.L + 1, self.F))
        self.state = np.vstack((np.clip(self.state[0:self.L, :] + actions.reshape((self.L, self.F)), 0, 1),
                                self.req.reshape((1, self.F)) / self.req.sum()))
        r,delays = self.reward(self.state[0:self.L, :]) #奖励函数和延迟
        self.state = self.state.reshape(-1)
        r = float(r)
        delays = float(delays) #延迟
        info = {'delay':delays}
        return self.state.astype(np.float32), r, False, info

    def reset(self): #环境的重置
        self.Zipf()
        self.request()
        self.state = np.random.rand(self.L, self.F)
        for i in range(self.L):
            self.state[i, :] = self.state[i, :] / self.state[i, :].sum()
        req = self.req.reshape((1, self.F)) / self.req.sum()
        state0 = np.vstack((self.state, req))
        self.state = state0.reshape(-1)
        return self.state.astype(np.float32)

    #根据成功率动态调整alpha：
    def adjust_alpha_based_on_success_rate(self):
        success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 1
        base_alpha = 0.85           #0.8
        threshold = 0.85          # 设置成功率阈值
        adjustment_factor = 0.05  # alpha的调整幅度

        if success_rate < threshold:
            alpha = base_alpha + adjustment_factor
        else:
            alpha = base_alpha

        return alpha

    def close(self):
        pass
