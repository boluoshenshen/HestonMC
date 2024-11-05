import numpy as np

class HestonModel:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        初始化 Heston 模型参数
        :param S0: 初始资产价格
        :param v0: 初始波动率
        :param kappa: 均值回复速率
        :param theta: 长期均值
        :param sigma: 波动率的波动率
        :param rho: 资产价格和波动率之间的相关性
        :param r: 无风险利率
        """
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r

    def simulate_path(self, scheme, T, num_steps):
        """
        使用给定的离散化方案生成路径
        :param scheme: 离散化方案实例（如 EulerScheme, QEScheme）
        :param T: 到期时间
        :param num_steps: 时间步长数
        :return: 资产价格和波动率路径
        """
        dt = T / num_steps  # 计算每个时间步长
        S, v = self.S0, self.v0  # 初始价格和波动率
        S_path, v_path = [S], [v]  # 存储价格和波动率路径

        for _ in range(num_steps):
            S, v = scheme.step(S, v, dt)  # 使用离散化方案更新 S 和 v
            S_path.append(S)
            v_path.append(v)

        return np.array(S_path), np.array(v_path)
