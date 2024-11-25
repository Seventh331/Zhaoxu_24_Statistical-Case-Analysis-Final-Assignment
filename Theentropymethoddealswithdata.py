import pandas as pd
f = open('核心期刊数据.csv')
df = pd.read_csv(f)
df = df.dropna().reset_index(drop=True)
#如果编辑解释器不是jupyter-notebook或命令行，用print函数包裹下面的代码
df.head()


class EmtropyMethod:
    def __init__(self, index, positive, negative, row_name):
        if len(index) != len(row_name):
            raise Exception('数据指标行数与行名称数不符')
        if sorted(index.columns) != sorted(positive + negative):
            raise Exception('正项指标加负向指标不等于数据指标的条目数')

        self.index = index.copy().astype('float64')
        self.positive = positive
        self.negative = negative
        self.row_name = row_name.copy()

def uniform(self):
    uniform_mat = self.index.copy()
    min_index = {column: min(uniform_mat[column]) for column in uniform_mat.columns}
    max_index = {column: max(uniform_mat[column]) for column in uniform_mat.columns}
    for i in range(len(uniform_mat)):
        for column in uniform_mat.columns:
        if column in self.negative:
        uniform_mat[column][i] = (uniform_mat[column][i] - min_index[column]) / (
                max_index[column] - min_index[column])
        else:
        uniform_mat[column][i] = (max_index[column] - uniform_mat[column][i]) / (
                max_index[column] - min_index[column])

    self.uniform_mat = uniform_mat
    return self.uniform_mat

def cal_finalscore()
   def calc_probability(self):
       try:
                p_mat = self.uniform_mat.copy()
            except AttributeError:
                raise Exception('你还没进行归一化处理，请先调用uniform方法')
            for column in p_mat.columns:
                sigma_x_1_n_j = sum(p_mat[column])
                # 为了取对数计算时不出现无穷,将比重为0的值修改为1e-6
                p_mat[column] = p_mat[column].apply(
                    lambda x_i_j: x_i_j / sigma_x_1_n_j if x_i_j / sigma_x_1_n_j != 0 else 1e-6)

            self.p_mat = p_mat
    return p_mat

 def calc_emtropy(self):
        try:
            self.p_mat.head(0)
        except AttributeError:
            raise Exception('你还没计算比重，请先调用calc_probability方法')

        import numpy as np
        e_j = -(1 / np.log(len(self.p_mat) + 1)) * np.array(
            [sum([pij * np.log(pij) for pij in self.p_mat[column]]) for column in self.p_mat.columns])
        ejs = pd.Series(e_j, index=self.p_mat.columns, name='指标的熵值')

        self.emtropy_series = ejs
        return self.emtropy_series

    def calc_emtropy_redundancy(self):
        try:
            self.d_series = 1 - self.emtropy_series
            self.d_series.name = '信息熵冗余度'
        except AttributeError:
            raise Exception('你还没计算信息熵，请先调用calc_emtropy方法')

        return self.d_series

    def calc_Weight(self):
        self.uniform()
        self.calc_probability()
        self.calc_emtropy()
        self.calc_emtropy_redundancy()
        self.Weight = self.d_series / sum(self.d_series)
        self.Weight.name = '权值'
        return self.Weight

    # 归一化矩阵
    indexs = ['出版文献量', '综合影响因子', '复合影响因子', '基金论文比', '篇均他引', '篇均被引', 'h指数', '引用刊数',
              '总被引频次', '篇均引文数']
    positive = indexs
    negative = []
    journal_name = df['期刊名']
    index = df[indexs]
    em = EmtropyMethod(index, negative, positive, journal_name)
    em.uniform()
    # 计算每个值在指标所有值中的比重
    em.calc_probability()
    # 计算指标的熵
    em.calc_emtropy()
    # 计算信息熵冗余度(1-熵值)
    em.calc_emtropy_redundancy()
    # 计算权重(将信息熵冗余度归一化)
    em.calc_Weight()
    em_weight = [0.115301, 0.157278, 0.143768, 0.083407, 0.077019, 0.089817, 0.051464, 0.093990, 0.120923, 0.067034]
    F = [[-0.066, 0.280, -0.022],
         [0.216, -0.012, -0.020],
         [0.209, -0.003, 0.030],
         [-0.068, 0.029, 0.646],
         [0.219, 0.006, -0.088],
         [0.218, 0.000, -0.050],
         [0.115, 0.244, -0.221],
         [-0.019, 0.276, -0.057],
         [0.003, 0.269, 0.034],
         [0.107, -0.084, 0.458]]
    weight = [0.47123, 0.3137, 0.13607]

import numpy as np
weight = np.array(weight)
indicator_weight = []
for i in range(10):
    em_indicator = em_weight[i]
    Fi = np.array(F[i])
    weight_i = em_indicator * np.dot(Fi, weight)
    indicator_weight.append(weight_i)


# 计算指标的最终权重
ejs = pd.Series(indicator_weight, index=indexs, name='指标的熵值')
ejs

writer = pd.ExcelWriter('normalized.xlsx')
em.uniform().to_excel(writer, 'Sheet1')
writer.save()