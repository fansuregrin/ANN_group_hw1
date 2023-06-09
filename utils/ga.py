"""Credit of this code: https://github.com/cattidea/bp-ga-pytorch"""
import numpy as np
import copy
import torch


def params_to_chrom(params):
    """ 模型参数转码为染色体 """
    chrom = np.empty(0)
    for key in params:
        chrom = np.append(chrom, params[key].cpu().numpy().flatten(), axis=-1)
    return chrom

def chrom_to_params(chrom, params_template, device):
    """ 染色体转码为模型参数（需参数模版） """
    params = copy.deepcopy(params_template)
    idx = 0
    for key in params:
        param_length = np.prod(params_template[key].shape)
        param = torch.from_numpy(chrom[idx: idx+param_length].reshape(params_template[key].shape)).to(device)
        params[key] = param
        idx += param_length
    return params


class Genome():
    """ 种群基因组，用于管理整个种群的染色体，数据分布为 U(0, 1)

    Args:
        pop_size: 种群规模
        chrom_len: 染色体长度

    Attributes:
        pop_size: 种群规模
        chrom_len: 染色体长度
        data: 基因组数据
        best: 最佳染色体
    """

    def __init__(self, pop_size, chrom_len):
        """ 初始化种群 """
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.data = None
        self.best = None

    def select(self, fitness_array):
        """ 选择 """
        raise NotImplementedError()

    def cross(self, cross_prob):
        """ 交叉 """
        raise NotImplementedError()

    def mutate(self, mutate_prob, progress):
        """ 突变 """
        raise NotImplementedError()

    def __getitem__(self, index):
        """ 直接获取内部数据 """
        return self.data[index].copy()

    def __setitem__(self, index, value):
        """ 直接修改内部数据 """
        self.data[index] = value.copy()

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式，分布仍为 U(0, 1) """
        raise NotImplementedError()

    def view(self, index, bound):
        """ 获取某一项编码数据的真实分布数据 """
        chrom = self._to_view(self.data[index])
        return (bound[1] - bound[0]) * chrom + bound[0]

    def view_best(self, bound):
        """ 获取最佳编码数据的真实分布数据 """
        chrom = self._to_view(self.best)
        return (bound[1] - bound[0]) * chrom + bound[0]


class GenomeReal(Genome):
    """ 实值编码基因组 """

    def __init__(self, pop_size, chrom_len):
        super().__init__(pop_size, chrom_len)
        self.data = np.random.uniform(0, 1, size=(pop_size, chrom_len))

    def select(self, fitness_array):
        """ 选择 """
        indices = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data[:], fitness_array[:] = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
        """ 交叉 """
        for idx in range(self.pop_size):
            if np.random.rand() < cross_prob:
                idx_other = np.random.choice(
                    np.delete(np.arange(self.pop_size), idx), size=1)
                cross_points = np.random.random(
                    self.chrom_len) < np.random.rand()
                cross_rate = np.random.rand()
                self.data[idx, cross_points], self.data[idx_other, cross_points] = \
                    (1-cross_rate) * self.data[idx, cross_points] + cross_rate * self.data[idx_other, cross_points], \
                    (1-cross_rate) * self.data[idx_other, cross_points] + \
                    cross_rate * self.data[idx, cross_points]

    def mutate(self, mutate_prob, progress):
        """ 突变 """
        for idx in range(self.pop_size):
            if np.random.rand() < mutate_prob:
                mutate_position = np.random.choice(
                    np.arange(self.chrom_len), size=1)
                self.data[idx][mutate_position] += np.random.uniform(0, 1) * (
                    np.random.randint(2)-self.data[idx][mutate_position]) * (1-progress)**2

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式，分布仍为 U(0, 1) """
        return chrom


class GenomeBinary(Genome):
    """ 二进制编码基因组 """

    def __init__(self, pop_size, chrom_len, code_len=16):
        super().__init__(pop_size, chrom_len)
        self.code_len = code_len
        self.data = np.random.random((pop_size, chrom_len*code_len)) < 0.5
        self.binary_template = np.zeros(code_len)
        for i in range(code_len):
            self.binary_template[i] = (2**i) / 2**code_len

    def select(self, fitness_array):
        """ 选择 """
        indices = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data[:], fitness_array[:] = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
        """ 交叉 """
        for idx in range(self.pop_size):
            if np.random.rand() < cross_prob:
                idx_other = np.random.choice(
                    np.delete(np.arange(self.pop_size), idx), size=1)
                cross_points = np.random.random(
                    self.chrom_len*self.code_len) < np.random.rand()
                self.data[idx, cross_points], self.data[idx_other, cross_points] = \
                    self.data[idx_other,
                              cross_points], self.data[idx, cross_points]

    def mutate(self, mutate_prob, progress):
        """ 突变 """
        for idx in range(self.pop_size):
            if np.random.rand() < mutate_prob:
                mutate_position = np.random.choice(
                    np.arange(self.chrom_len*self.code_len), size=1)
                self.data[idx][mutate_position] = ~self.data[idx][mutate_position]

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式，分布仍为 U(0, 1) """
        return np.sum(chrom.reshape(self.chrom_len, self.code_len) * self.binary_template, axis=-1)


class GA():
    """ 遗传算法
    Args:
        pop_size: 种群规模
        chrom_len: 染色体长度
        bound: 染色体真实上下界
        calculate_fitness_func: 用于计算适应度的函数
        GenomeClass: 基因组所使用的编码（实值编码：GenomeReal，二进制编码：GenomeBinary）
        cross_prob: 交叉概率
        mutate_prob: 变异概率

    Attributes:
        pop_size: 种群规模
        chrom_len: 染色体长度
        bound: 染色体真实上下界
        cross_prob: 交叉概率
        mutate_prob: 变异概率
        genome: 种群基因组（用于管理全部染色体）
        fitness_array: 适应度数组
        best_fitness: 最佳适应度

    Examples:
        >>> pop_size = 10
        >>> chrom_len = 100
        >>> bound = [0, 1] # np.vstack([np.zeros(chrom_len), np.ones(chrom_len)])
        >>> def calculate_fitness(x):
        >>>     # 根据染色体真实数据计算适应度
        >>>     return ...
        >>> ga = GA(pop_size, chrom_len, bound, calculate_fitness, GenomeClass=GenomeBinary, cross_prob=0.8, mutate_prob=0.3)
        >>> ga.genetic(1000, log=True)
        >>> ga.result()
    """

    def __init__(self, pop_size, chrom_len, bound, calculate_fitness_func,
                 GenomeClass=GenomeReal, cross_prob=0.8, mutate_prob=0.03):
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.bound = bound
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.calculate_fitness = calculate_fitness_func

        self.fitness_array = np.zeros(pop_size)
        self.genome = GenomeClass(pop_size=pop_size, chrom_len=chrom_len)
        self.update_fitness()
        self.best_fitness = 0
        self.update_records()

    def update_records(self):
        """ 更新最佳记录 """
        best_index = np.argmax(self.fitness_array)
        self.genome.best = self.genome[best_index]
        self.best_fitness = self.fitness_array[best_index]

    def replace(self):
        """ 使用前代最佳替换本代最差 """
        worst_index = np.argmin(self.fitness_array)
        self.genome[worst_index] = self.genome.best
        self.fitness_array[worst_index] = self.best_fitness

    def update_fitness(self):
        """ 重新计算适应度 """
        for idx in range(self.pop_size):
            self.fitness_array[idx] = self.calculate_fitness(
                self.genome.view(idx, self.bound))

    def genetic(self, num_gen, logger=None):
        """ 开始运行遗传算法 """
        for e in range(num_gen):
            self.genome.select(self.fitness_array)
            self.genome.cross(self.cross_prob)
            self.genome.mutate(self.mutate_prob, progress=e/num_gen)
            self.update_fitness()

            if self.fitness_array[np.argmax(self.fitness_array)] > self.best_fitness:
                self.replace()
                self.update_records()

            if logger is not None:
                logger.info('Evolution {} Best: {}, Average: {}'.format(
                    e+1, self.fitness_array.max(), self.fitness_array.mean()))

    def result(self):
        """ 输出最佳染色体 """
        return self.genome.view_best(self.bound)


if __name__ == '__main__':
    def F(x):
        return x + 10*np.sin(5*x) + 7*np.cos(4*x)

    def calculate_fitness(x):
        return F(x) + 50

    POP_SIZE = 5
    CHROM_LEN = 1
    X_BOUND = (-2, 5)
    bound = np.zeros((2, CHROM_LEN))
    bound[0] = X_BOUND[0] * np.ones(CHROM_LEN)
    bound[1] = X_BOUND[1] * np.ones(CHROM_LEN)

    # Evolution
    ga = GA(POP_SIZE, CHROM_LEN, bound, calculate_fitness,
            GenomeClass=GenomeBinary, cross_prob=0.8, mutate_prob=0.3)
    ga.genetic(1000, log=True)

    # Plot
    import matplotlib.pyplot as plt
    x_axis = np.linspace(*X_BOUND, 200)
    print("Best fitness: {}, target: {}".format(calculate_fitness(
        ga.result())[0], calculate_fitness(x_axis[np.argmax(F(x_axis))])))
    plt.plot(x_axis, F(x_axis))
    plt.scatter(ga.result(), F(ga.result()), color='r')
    plt.show()
