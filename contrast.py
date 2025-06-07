'''
用于模拟和评估不同缓存策略在一个假设的通信网络环境中的性能。
代码的目的是对比两种缓存策略：基于 CVXPY 的凸优化方法（CVX）和基于 eta 的方法。
论文中提到的HCO启发式凸优化算法就是该优化算法
'''

import cvxpy as cp
import numpy as np
from environment import highway_cf
from pathlib import Path
import scipy.io as scio
import time
import os
import warnings

class Datas:
    def __init__(self, K, F, L, cap, dis, eps,alpha):
        self.env = highway_cf(AP=L, UE=K, Cap=cap, dis=dis, files=F, epsilon=eps,alpha=alpha)


        self.buffer_cvx = buffer(max_size=500, AP=L, UE=K, files=F)
        self.buffer_eta = buffer(max_size=500, AP=L, UE=K, files=F)
        self.buffer_fcfs = buffer(max_size=500, AP=L, UE=K, files=F)
        self.buffer_lru = buffer(max_size=500, AP=L, UE=K, files=F)

        self.avr_req_cvx = 0
        self.avr_req_eta = 0
        self.avr_fcfs = 0
        self.avr_lru = 0

        self.cvx = CVX(K, F, L, cap, self.env.file_size)
        self.qifa = eta(K, F, L, cap, self.env.file_size)
        self.fcfs = FCFS(UE=K, file=F, AP=L, Cap=cap, file_size=self.env.file_size)
        self.lru = lru(UE=K, file=F, AP=L, Cap=cap, file_size=self.env.file_size)

        self.cvx_satisfy = np.ones((K, 1))
        self.qifa_satisfy = np.ones((K, 1))
        self.fcfs_satisfy = np.ones((K, 1))
        self.lru_satisfy = np.ones((K, 1))

        self.cvx_weight = np.zeros((F, 1))
        self.qifa_weight = np.zeros((F, 1))
        self.fcfs_weight = np.zeros((F, 1))
        self.lru_weight = np.zeros((F, 1))


        self.req_cvx = np.zeros((1, F))
        self.req_eta = np.zeros((1, F))
        self.req_fcfs = np.zeros((1, F))
        self.req_lru = np.zeros((1, F))

        self.avr_req_cvx = 0
        self.avr_req_eta = 0
        self.avr_req_fcfs = 0
        self.avr_req_lru = 0

        self.need_cvx = np.zeros(K)
        self.need_eta = np.zeros(K)
        self.need_fcfs = np.zeros(K)
        self.need_lru = np.zeros(K)


    def contrast(self, cap, file, AP, dis, K):
        self.arv_req()

        policy_cvx = self.cvx.simulation(self.avr_req_cvx)
        policy_qifa = self.qifa.simulation(self.avr_req_eta)
        policy_fcfs = self.fcfs.simulation(self.avr_req_fcfs)  # 使用FCFS策略

        policy_lru = self.lru.simulation(self.avr_req_lru)     # 使用LRU策略
        self.lru.update_file_usage()  # 更新文件使用情况

        self.env.weight = self.cvx_weight
        self.env.satisfy = self.cvx_satisfy
        self.env.request()
        self.req_cvx = self.env.req
        self.need_cvx = self.env.need
        r_cvx = self.cvx.reward(policy_cvx, self.req_cvx, self.cvx_weight)
        QOE_cvx, self.cvx_satisfy, self.cvx_weight,cvx_delay = tau_QoE(cache=policy_cvx, req=self.req_cvx, cap=cap,
                                                                      file=file, AP=AP, file_size=self.env.file_size,
                                                                      dis=dis, need=self.need_cvx,
                                                                      weight=self.cvx_weight, UE=K)
        self.buffer_cvx.store_transition(self.req_cvx)

        # 使用FCFS策略计算奖励和QoE
        self.env.weight = self.fcfs_weight
        self.env.satisfy = self.fcfs_satisfy
        self.env.request()
        self.req_fcfs = self.env.req
        self.need_fcfs = self.env.need
        r_fcfs = self.fcfs.reward(policy_fcfs, self.req_fcfs, self.fcfs_weight)
        QOE_fcfs, self.fcfs_satisfy, self.fcfs_weight,fcfs_delay = tau_QoE(cache=policy_fcfs, req=self.req_fcfs, cap=cap,
                                                                file=file, AP=AP,
                                                                file_size=self.env.file_size,
                                                                dis=dis, need=self.need_fcfs,
                                                                weight=self.fcfs_weight, UE=K)
        self.buffer_fcfs.store_transition(self.req_fcfs)

        # 使用LRU策略计算奖励和QoE
        self.env.weight = self.lru_weight
        self.env.satisfy = self.lru_satisfy
        self.env.request()
        self.req_lru = self.env.req
        self.need_lru = self.env.need
        r_lru = self.lru.reward(policy_lru, self.req_lru, self.lru_weight)

        QOE_lru, self.lru_satisfy, self.lru_weight,lru_delay = tau_QoE(cache=policy_lru, req=self.req_lru, cap=cap,
                                                                file=file, AP=AP,
                                                                file_size=self.env.file_size,
                                                                dis=dis, need=self.need_lru,
                                                                weight=self.lru_weight, UE=K)
        self.buffer_lru.store_transition(self.req_lru)


        # 使用eta策略计算奖励和QoE
        self.env.weight = self.qifa_weight
        self.env.satisfy = self.qifa_satisfy
        self.env.request()
        self.req_eta = self.env.req
        self.need_eta = self.env.need

        r_qifa = self.qifa.reward(policy_qifa, self.req_eta, self.qifa_weight)
        QOE_qifa, self.qifa_satisfy, self.qifa_weight,qifa_delay = tau_QoE(cache=policy_qifa, req=self.req_eta, cap=cap,
                                                                          file=file, AP=AP,
                                                                          file_size=self.env.file_size,
                                                                          dis=dis, need=self.need_eta,
                                                                          weight=self.qifa_weight, UE=K)
        self.buffer_eta.store_transition(self.req_eta)
        # 返回cvx、qifa、fcfs、lru的奖励和QoE
        return r_cvx, QOE_cvx, cvx_delay, \
            r_qifa, QOE_qifa, qifa_delay, \
            r_fcfs, QOE_fcfs, fcfs_delay, \
            r_lru, QOE_lru, lru_delay



    #contrast_time用来计算cvx的运行时间，在time.py中调用和PPO算法进行对比
    #后期可以在这里加入其他算法的时间对比，比如eta，fcfs，lru
    def contrast_time(self, cap, file, AP, dis, K):
        time_start = time.time()                            #开始
        self.arv_req()                                      #更新请求统计
        policy_cvx = self.cvx.simulation(self.avr_req_cvx)  #基于凸优化的策略模拟
        time_end = time.time()
        times = time_start - time_end

        self.env.weight = self.cvx_weight
        self.env.satisfy = self.cvx_satisfy
        self.env.request()
        self.req_cvx = self.env.req
        self.need_cvx = self.env.need

        r_cvx = self.cvx.reward(policy_cvx, self.req_cvx, self.cvx_weight)

        # 使用凸优化策略计算奖励和QoE
        QOE_cvx, self.cvx_satisfy, self.cvx_weight,total_delay = tau_QoE(cache=policy_cvx, req=self.req_cvx, cap=cap,
                                                             file=file, AP=AP, file_size=self.env.file_size,
                                                             dis=dis, need=self.need_cvx,
                                                             weight=self.cvx_weight, UE=K)
        self.buffer_cvx.store_transition(self.req_cvx)
        return r_cvx, QOE_cvx, times

    def arv_req(self):
        if self.buffer_cvx.mem_count == 0:
            self.env.request()
            self.avr_req_cvx = self.env.req
            self.avr_req_eta = self.env.req
            self.avr_req_fcfs = self.env.req  # 添加FCFS的请求统计更新
            self.avr_req_lru = self.env.req  # 添加LRU的请求统计更新
        else:
            if self.buffer_cvx.mem_count <= self.buffer_cvx.mem_size:
                self.avr_req_cvx = self.buffer_cvx.req_his[0:self.buffer_cvx.mem_count].mean(axis=0)
                self.avr_req_eta = self.buffer_eta.req_his[0:self.buffer_cvx.mem_count].mean(axis=0)
                self.avr_req_fcfs = self.buffer_fcfs.req_his[0:self.buffer_fcfs.mem_count].mean(axis=0)  # 添加FCFS的请求统计更新
                self.avr_req_lru = self.buffer_lru.req_his[0:self.buffer_lru.mem_count].mean(axis=0)  # 添加LRU的请求统计更新
            else:
                self.avr_req_cvx = self.buffer_cvx.req_his.mean(axis=0)
                self.avr_req_eta = self.buffer_eta.req_his.mean(axis=0)
                self.avr_req_fcfs = self.buffer_fcfs.req_his.mean(axis=0)  # 添加FCFS的请求统计更新
                self.avr_req_lru = self.buffer_lru.req_his.mean(axis=0)   # 添加LRU的请求统计更新

class buffer:
    def __init__(self, max_size, AP, files, UE):
        self.mem_size = max_size
        self.mem_count = 0

        self.L = AP
        self.F = files
        self.K = UE

        self.req_his = np.zeros((self.mem_size, self.F, 1))

    def store_transition(self, req):
        index = self.mem_count % self.mem_size

        self.req_his[index] = req
        self.mem_count += 1

#不同于environment.py的hit_2 这个考虑了权重，用在了cvx和eta 计算命中率
def hit_3(cache, F, cap, file_size, req, K, weight):
    r = 0
    #这里hit调用的a值有问题
    # print(f"reslut:{np.sum(cache, axis=0).reshape((F, 1)) * cap / file_size}")
    # print(f"np.ones_like(file_size):{np.ones_like(file_size)}")
    a = np.minimum(np.sum(cache, axis=0).reshape((F, 1)) * cap / file_size, np.ones_like(file_size))
    # print(f"a:{a}")
    for f in range(F):

        r = r + (req[f] + weight[f]) * a[f]
    return r / (req + weight).sum()



def tau_QoE(cache, req, cap, file, AP, file_size, dis, weight, need, UE):
    SE = 200                                 #后续在调整QoE加入延迟考虑 应该调整这个SE 使得延迟敏感提高
    satisfy = np.zeros((UE, 1))
    T = np.array([1, 20])
    taus = np.zeros((AP, file))              #计算每个文件在每个接入点AP的传输延迟
    #np.sum(cache, axis=0).reshape((file, 1) 取cache第0轴求和，得到每个文件在所有接入点的总缓存量，再reshape成一个列向量
    a = np.maximum(file_size - np.sum(cache, axis=0).reshape((file, 1)) * cap, np.zeros_like(file_size)) #计算每个文件中未被缓存的部分 并计算基本延迟
    for i in range(file):                   #循环便利每个文件 对于每个AP 计算距离和缓存状态的延迟 更新taus数组
        taus[:, i] = -a[i] * T[1]           #a[i]是文件i未被缓存的大小 T[]是延迟参数，延迟和未缓存的数据量成反比
        for j in range(AP):
            distance = dis * abs(np.arange(AP) - j)         #计算从当前接入点AP到其他接入点的距离，dis是距离参数，-j表示计算每个接入点到当前接入点j的距离绝对值
            for num in range(AP):                           #遍历可能缓存的接入点
                b = int(min(AP - 1, j + num))               #b考虑缓存片段的结束索引
                c = int(max(0, j - num))                    #c考虑缓存片段的开始索引
                if cache[c:b + 1, i].sum() * cap + a[i] >= file_size[i]:    #计算当前接入点j是否有足够片段来存入缓存
                    taus[j, i] = taus[j, i] - np.max(cache[c:b + 1, i] * distance[c:b + 1]) * cap * T[0]   #文件i在接入点j的传输延迟-接入点j到c-b之间最远的缓存片段延迟
                    break

    single = np.zeros((file, 1))             #文件的最小传输延迟
    file_delay = np.zeros((file, 1))
    #存储每个文件的最小传输延迟
    for i in range(file):
        single[i] = np.min(taus[:, i])


    t = 0
    QoE = 0
    alpha = 0.8
    middle = -(weight + req).reshape(-1)   #文件权重和请求次数向量相加
    line = np.argsort(middle)              #按照middle数组升序，可以率先考虑请求次数更多或者权重更大的文件
    delay = 0                              #自定义奖励函数的延迟权重因子
    for i in line:                         #迭代处理每个文件的请求 更新QoE
        t = t + single[i]
        #1/SE 每个请求单位对延迟的敏感度 SE越大表示系统对延迟的敏感度较低
        #后续在调整QoE加入延迟考虑 应该调整这个SE 使得延迟敏感提高
        t = t - req[i] * (1 / SE)

        if t > -500:
            QoE = QoE + req[i] + weight[i]                  #源码
            satisfy[np.where(need == i)] = 1                #satify记录请求是否满足
            weight[i] = 0
            # delay = delay + single[i] * (req[i] + weight[i])  # 加入了延迟权重项作为奖励函数的一部分，延迟和文件传输延迟、请求次数、文件权重相关
            delay = delay + single[i]
        else:
            weight[i] = weight[i] + req[i]                   #这段请求未能在可接受的延迟内得到满足，增加文件权重，提高缓存满足概率概率
            #对于未满足的请求，延迟应该更大，考虑这个文件的请求次数、文件权重、大小
            delay_pro = weight[i] + req[i] + file_size[i]
            delay = delay + delay_pro

    # return np.array(QoE/((weight + req).sum())), satisfy, weight #源码
    total_delay = np.sum(delay)
    return np.array( alpha * QoE / ((weight + req).sum()) + (1-alpha)*delay/(req+weight+file_size+single).sum() ), satisfy, weight, total_delay


class eta:
    def __init__(self, UE, file, AP, Cap, file_size):
        self.K = UE
        self.F = file
        self.L = AP
        self.cap = Cap
        self.file_size = file_size
        self.policy = np.zeros((self.L, self.F))

    def simulation(self, req):
        req = req / req.sum()
        for f in range(self.F):
            self.policy[:, f] = req[f]
        return self.policy

    def reward(self, cache, req, weight):
        r_pro = hit_3(cache=cache, F=self.F, cap=self.cap, file_size=self.file_size, req=req, K=self.K, weight=weight)
        return r_pro

class CVX:
    def __init__(self, UE, file, AP, Cap, file_size):
        self.K = UE
        self.L = AP
        self.F = file
        self.cap = Cap
        self.file_size = file_size
        self.sumf = np.ones((self.F, 1))
        self.suml = np.ones((1, self.L))
        self.c = np.ones((self.L, 1))
        self.req = 0
        self.one = np.eye(self.F)
        self.zero = np.zeros((self.L, self.F))

    def simulation(self, req):
        self.req = req
        cache = cp.Variable((self.L, self.F))
        target = cp.matmul(self.req.T,
                           cp.minimum(cp.multiply(cp.matmul(self.suml, cache).T, self.cap / self.file_size), 1))
        prob = cp.Problem(cp.Maximize(target), [cache @ self.sumf <= self.c, cache @ self.one >= self.zero])
        prob.solve()

        return cache.value

    def reward(self, cache, req, weight):
        r = hit_3(cache=cache, F=self.F, cap=self.cap, file_size=self.file_size, req=req, K=self.K, weight=weight)
        return r
'''
根据FCFS思想设计缓存策略
'''
class FCFS:
    def __init__(self, UE, file, AP, Cap, file_size):
        self.K = UE  # 用户设备数量
        self.F = file  # 文件数量
        self.L = AP  # 接入点数量
        self.cap = Cap  # 缓存容量
        self.file_size = file_size  # 文件大小数组
        self.cache = np.zeros((self.L, self.F))  # 初始化缓存矩阵
        self.cache_order = []  # 缓存顺序列表

        self.req = np.zeros((self.K, self.F))


    def add_to_cache(self, file_index, AP_index):
        # 添加文件到缓存
        if len(self.cache_order) >= self.cap:
            old_file_index = self.cache_order.pop(0)  # 移除最老的文件索引
            self.cache[AP_index, old_file_index] = 0
        self.cache[AP_index, file_index] = 1
        self.cache_order.append(file_index)

    def simulation(self, req):
        #模拟整个缓存过程，如果有请求 调用缓存方法加入到缓存中
        self.req = req
        # 根据请求更新缓存策略
        for f in range(self.F):  # 遍历所有文件
            if req[f] > 0:  # 如果文件有请求
                for l in range(self.L):  # 遍历所有接入点
                    if self.cache[l, f] == 0:  # 如果文件不在缓存中
                        self.add_to_cache(f, l)

        return self.cache

    def reward(self, cache,req, weight):
        #用hit_3计算命中率
        # print(cache.flatten()[:10])  # 打印cache的前50个元素
        # print(req.flatten()[:10])    # 打印req的前50个元素
        r = hit_3(cache=cache, F=self.F, cap=self.cap, file_size=self.file_size, req=req, K=self.K, weight=weight)
        return r

# # 使用FCFS类
# fifo = FCFS(UE=300, file=20, AP=15, Cap=6, file_size=[文件大小数组])
# fifo.simulation(req)  # req 是请求矩阵
# reward = fifo.reward(req, weight)  # weight 是权重向量

#按照LRU思想设计缓存策略，最近最少使用进行缓存替换
class lru:
    def __init__(self, UE, file, AP, Cap, file_size):
        self.K = UE     # 用户设备数量
        self.F = file   # 文件数量
        self.L = AP     # 接入点
        self.cap = Cap  # 缓存容量
        self.file_size = file_size               # 文件大小数组
        self.cache = np.zeros((self.L, self.F))  # 初始化缓存矩阵
        self.file_usage = {}                     # 文件使用情况的记录

        self.req = 0

    def simulation(self, req):
        self.req = req
        # 根据请求更新缓存策略
        for f in range(self.F):  # 遍历所有文件
            if req[f] > 0:  # 如果文件有请求
                for l in range(self.L):  # 遍历所有接入点
                    if self.cache[l, f] == 0:  # 如果文件不在缓存中
                        self.add_to_cache(f, l)

        return self.cache

    def reward(self, cache,req, weight):
        r = hit_3(cache=self.cache, F=self.F, cap=self.cap, file_size=self.file_size, req=self.req, K=self.K, weight=weight)
        # 计算命中率
        return r

    def add_to_cache(self, file_index, AP_index):
        # 添加文件到缓存
        if len(self.file_usage) >= self.cap:
            least_used_file = min(self.file_usage, key=self.file_usage.get)  # 找到最近最少使用的文件
            self.cache[AP_index, least_used_file] = 0  # 从缓存中移除最近最少使用的文件
            del self.file_usage[least_used_file]        # 从文件使用情况的记录中移除最近最少使用的文件
        self.cache[AP_index, file_index] = 1              # 将新的文件添加到缓存中
        self.file_usage[file_index] = 0                 # 将新的文件添加到文件使用情况的记录中

    #在每次请求后，需要调用update_file_usage方法来更新文件使用情况的记录
    def update_file_usage(self):
        # 更新文件使用情况的记录
        for file_index in self.file_usage:
            self.file_usage[file_index] += 1


if __name__ == "__main__":
    L= 15   #15
    K = 300 #300
    Cap = 6
    F = 40  #20
    dis = 10
    eps = 1.1
    base_alpha = 0.8
    datas = Datas(K=K, F=F, L=L, cap=Cap, dis=dis, eps=eps,alpha=base_alpha)

    hit_C = np.zeros(10000)
    QoE_C = np.zeros(10000)
    delay_C = np.zeros(10000)

    hit_E = np.zeros(10000)
    QoE_E = np.zeros(10000)
    delay_E = np.zeros(10000)

    hit_F = np.zeros(10000)
    QoE_F = np.zeros(10000)
    delay_F = np.zeros(10000)

    hit_L = np.zeros(10000)
    QoE_L = np.zeros(10000)
    delay_L = np.zeros(10000)




    datas.env.Zipf()
    for i in range(10000): #10000
        hit_cvx, QOE_cvx,cvx_delay, \
            hit_eta, QOE_eta,qifa_delay, \
            hit_fcfs, QOE_fcfs,fcfs_delay, \
            hit_lru, QoE_lru,lru_delay\
            = datas.contrast(cap=Cap, file=F, AP=L, K=K, dis=dis)

        hit_C[i] = hit_cvx
        QoE_C[i] = QOE_cvx
        delay_C[i] = cvx_delay

        hit_E[i] = hit_eta
        QoE_E[i] = QOE_eta
        delay_E[i] = qifa_delay

        hit_F[i] = hit_fcfs
        QoE_F[i] = QOE_fcfs
        delay_F[i] = fcfs_delay

        hit_L[i] = hit_lru
        QoE_L[i] = QoE_lru
        delay_L[i] = lru_delay


    a = Path.cwd()
    #MAT_datas -> data
    # scio.savemat(a / "MAT_datas" / "cvx_hit.mat", {"cvx_hit": hit_C})
    # scio.savemat(a / "MAT_datas" / "cvx_qoe.mat", {"cvx_qoe": QoE_C})
    # scio.savemat(a / "MAT_datas" / "cvx_delay.mat", {"cvx_delay": delay_C})

    scio.savemat(a / "MAT_datas" / "eta_hit（F_40）.mat", {"eta_hit": hit_E})
    scio.savemat(a / "MAT_datas" / "eta_qoe（F_40）.mat", {"eta_qoe": QoE_E})
    scio.savemat(a / "MAT_datas" / "eta_delay（F_40）.mat", {"eta_delay": delay_E})

    # scio.savemat(a / "MAT_datas" / "fcfs_hit.mat", {"fcfs_hit": hit_F})
    # scio.savemat(a / "MAT_datas" / "fcfs_qoe.mat", {"fcfs_qoe": QoE_F})
    # scio.savemat(a / "MAT_datas" / "fcfs_delay.mat", {"fcfs_delay": delay_F})
    #
    # scio.savemat(a / "MAT_datas" / "lru_hit.mat", {"lru_hit": hit_L})
    # scio.savemat(a / "MAT_datas" / "lru_qoe.mat", {"lru_qoe": QoE_L})
    # scio.savemat(a / "MAT_datas" / "lru_delay.mat", {"lru_delay": delay_L})