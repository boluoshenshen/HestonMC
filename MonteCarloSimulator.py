import numpy as np
from HestonModel import HestonModel

class MonteCarloSimulator:
    def __init__(self, model, scheme, num_paths, T, num_steps):
        """
        初始化 Monte Carlo 模拟器
        :param model: HestonModel 实例
        :param scheme: 离散化方案实例（如 EulerScheme 或 QEScheme）
        :param num_paths: 模拟路径数
        :param T: 到期时间
        :param num_steps: 每条路径的时间步长数
        """
        self.model = model
        self.scheme = scheme
        self.num_paths = num_paths
        self.T = T
        self.num_steps = num_steps

    def generate_paths(self):
        """
        生成模拟路径
        :return: 资产价格路径和波动率路径矩阵（每一行表示一条路径）
        """
        S_paths = np.zeros((self.num_paths, self.num_steps + 1))
        v_paths = np.zeros((self.num_paths, self.num_steps + 1))

        for i in range(self.num_paths):
            S, v = self.model.S0, self.model.v0  # 每条路径的初始值
            S_paths[i, 0], v_paths[i, 0] = S, v

            for j in range(1, self.num_steps + 1):
                S, v = self.scheme.step(S, v, self.T / self.num_steps)
                S_paths[i, j] = S
                v_paths[i, j] = v

        return S_paths, v_paths

    def price_option(self, payoff_func):
        """
        使用 Monte Carlo 方法计算期权价格
        :param payoff_func: 期权的收益函数（如欧式看涨或看跌）
        :return: 期权价格
        """
        S_paths, _ = self.generate_paths()
        payoffs = np.maximum(payoff_func(S_paths[:, -1]), 0)
        discount_factor = np.exp(-self.model.r * self.T)
        return discount_factor * np.mean(payoffs)