import numpy as np
from MonteCarloSimulator import MonteCarloSimulator
from OptionPricer import OptionPricer
from HestonModel import HestonModel

class MethodComparison:
    def __init__(self, heston_model, T, K, true_price):
        """
        初始化 MethodComparison
        :param heston_model: HestonModel 实例
        :param T: 到期时间
        :param K: 期权执行价格
        :param true_price: 期权的理论真实价格（如半解析解）
        """
        self.heston_model = heston_model
        self.T = T
        self.K = K
        self.true_price = true_price

    def accuracy(self, scheme, num_paths, num_steps):
        """
        计算不同方法的准确度
        :param scheme: 离散化方案实例（如 EulerScheme 或 QEScheme）
        :param num_paths: 模拟路径数
        :param num_steps: 时间步长数
        :return: 准确度（与真实值的偏差）
        """
        simulator = MonteCarloSimulator(self.heston_model, scheme, num_paths, self.T, num_steps)
        option_pricer = OptionPricer(simulator)
        estimated_price = option_pricer.european_call(self.K)
        return abs(estimated_price - self.true_price)

    def convergence(self, scheme, num_paths, step_sizes):
        """
        计算不同方法的收敛性
        :param scheme: 离散化方案实例
        :param num_paths: 模拟路径数
        :param step_sizes: 一系列不同的时间步长
        :return: 收敛误差列表
        """
        errors = []
        for num_steps in step_sizes:
            error = self.accuracy(scheme, num_paths, num_steps)
            errors.append(error)
        return errors

    def stability(self, scheme, num_paths, num_steps, num_trials=10):
        """
        计算不同方法的稳定性
        :param scheme: 离散化方案实例
        :param num_paths: 模拟路径数
        :param num_steps: 时间步长数
        :param num_trials: 试验次数
        :return: 模拟结果的标准差（作为稳定性指标）
        """
        prices = []
        for _ in range(num_trials):
            simulator = MonteCarloSimulator(self.heston_model, scheme, num_paths, self.T, num_steps)
            option_pricer = OptionPricer(simulator)
            price = option_pricer.european_call(self.K)
            prices.append(price)
        return np.std(prices)
