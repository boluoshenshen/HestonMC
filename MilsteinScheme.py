import numpy as np
from HestonModel import HestonModel

class MilsteinScheme(HestonModel):
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        初始化 Milstein 离散化方案，并继承 HestonModel 参数
        """
        super().__init__(S0, v0, kappa, theta, sigma, rho, r)

    def step(self, S, v, dt):
        """
        使用 Milstein 离散化方案进行单步更新
        :param S: 当前资产价格
        :param v: 当前波动率
        :param dt: 时间步长
        :return: 下一个时间步的资产价格 S 和波动率 v
        """
        # 生成两个相关的正态随机变量
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  # 确保 Z1 和 Z2 的相关性为 rho

        # 使用 Milstein 方法更新波动率和资产价格
        # 更新波动率 v (确保非负)
        v_sqrt = np.sqrt(v)
        v_next = v + self.kappa * (self.theta - v) * dt + self.sigma * v_sqrt * np.sqrt(dt) * Z2 \
                 + 0.25 * self.sigma**2 * dt * (Z2**2 - 1)  # Milstein校正项
        v_next = max(v_next, 0)  # 确保波动率非负

        # 更新资产价格 S
        S_next = S * np.exp((self.r - 0.5 * v) * dt + v_sqrt * np.sqrt(dt) * Z1 \
                            + 0.5 * v_sqrt * self.sigma * dt * (Z1**2 - 1))  # Milstein校正项

        return S_next, v_next