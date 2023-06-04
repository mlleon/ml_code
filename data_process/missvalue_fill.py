"""
    非特殊情况不建议采用模型自带的填补方法。CART树的填补仅会导向训练集模型结果最优，不一定能真实的反应数据真实规律，
模型容易过拟合（cart树自带的填补方法只有原生算法才有，sklearn模块的CART树算法没有设置简单插补法这个步骤）；
LGB则是简单的把缺失值看成一类，实际上也就是在执行真值转化；
    无论何时，如果能够从业务上推断出缺失值的真实取值，都应该首先采用该值进行填补，这是最高优先级的填补策略没有之一；
缺失值的占比也会很大程度上影响缺失值填补策略的选择。缺失值占比过大或过小，都没有太大的填补意义。例如，当缺失值比例超过80%时，
几乎不可能从剩余的20%的数据中还原这一列应有的原始信息，无论用哪种方法填补，“瞎猜”的成分都比较大，最后的建模风险会非常高，
因此建议删除；而如果缺失值占比过小，例如仅有0.5%缺失值，无论用哪种方法填补，实际上数据的变动都不会对该特征的分布造成太大的影响，
因此填补的效果也非常有限，此时应优先考虑高效快速完成填补事宜；如果缺失值的比例为50%左右时，考虑是否缺失，如何填补缺失值可能会更有价值；
    很多时候，缺失值填补不一定仅仅停留在对当前列的缺失数据进行修改，很多时候可以采用不同填补方案、生成多个列，然后在实际进行特征筛选时再进行挑选，
甚至可以借助一些优化器筛选出最佳填补。
"""
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

# 设置显示DataFrame对象的所有行和列
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('max_colwidth', 100)  # value显示宽度100，默认为50
# --------------------------------------------------------#

# 1.给原始数据随机设置缺失值
names = ['怀孕次数', '2小时耐量实验值', '舒张压',
         '三头肌皮褶厚度', '2小时血清胰岛素浓度',
         'BMI', '遗传函数', 'age', '是否患病']

pima = pd.read_csv(r"E:\gitlocal\machine_learning\diabetes.csv", dtype='float')
pima.columns = names

# 1.1.确定插入缺失值的比例
missing_rate = 0.3

# 1.2.计算插入缺失值的点数
data = pima.copy()
x_full, y_full = data.iloc[:, :-1], data.iloc[:, -1]
n_samples, n_features = x_full.shape
# 30%的缺失率，需要有1843个缺失数据
n_missing_samples = int(np.round(n_samples * n_features * missing_rate))
# n_missing_samples  = int(np.round(n_samples * n_features * missing_rate))
# n_missing_samples  = int(np.floor(n_samples * n_features * missing_rate))
# np.round向上取整，返回.0格式的浮点数
# np.floor向下取整，返回.0格式的浮点数

# 1.3.获得插入缺失值的位置索引
rng = np.random.RandomState(0)
# 缺失值插入位置的列索引（0,8，1843）
missing_col_index = rng.randint(0, n_features, n_missing_samples)
# 缺失值插入位置的行索引（0,768，1843）
missing_row_index = rng.randint(0, n_samples, n_missing_samples)

# 1.4.按照对应的行索引和列索引插入缺失值到指定位置
x_missing = x_full.copy()
y_missing = y_full.copy()
# 按照索引数据（DataFrame）插入缺失值
for i in range(n_missing_samples):
    x_missing.iloc[missing_row_index[i], missing_col_index[i]] = np.nan
# 把标签列拼接回来，使数据集和原始数据集维度一致
data_missing = x_missing
data_missing["是否患病"] = y_missing


# --------------------------------------------------------#

def fill_method_contrast(data_df, colNames_fill_Methods):
    """
    缺失值填补方法

    :param data_df: 含有缺失值的数据集
    :param colNames_fill_Methods: 字典类型（key-缺失值特征列, value-缺失值填补方法）
                                  填补方法包括:
                                  常值（0,1...），均值('mean')，众数（'mode'）
                                  从缺失值后面的有效值从后向前填充('bfill', 'backfill')
                                  从缺失值前面的有效值从前向后填充('ffill', 'pad')
                                  使用差值函数插补缺失值：
                                  'nearest': 最近邻插值法,
                                  'zero': 阶梯差值,
                                  'slinear', 'linear': 线性差值,
                                  'quadratic', 'cubic'：2,3阶B样条曲线差值

    :return: 返回缺失值填补后的特征列组合的DataFrame对象
    """

    # 复制原始数据集和含有缺失值的数据集
    original_data = data_df.copy()
    missing_value_dF = data_df.copy()

    fill_method_Df = {}  # 用于存储填补后的特征列
    fillCols = []  # 存储填补后的特征列的列名

    # 填补方法字典，包含各种填补方法对应的操作
    fill_methods_dict = {
        'nearest': missing_value_dF.interpolate(method='nearest'),  # 最近邻插值法
        'zero': missing_value_dF.interpolate(method='zero'),  # 阶梯差值
        'slinear': missing_value_dF.interpolate(method='slinear'),  # 线性差值
        'linear': missing_value_dF.interpolate(method='linear'),  # 线性差值
        'quadratic': missing_value_dF.interpolate(method='quadratic'),  # 2阶B样条曲线差值
        'cubic': missing_value_dF.interpolate(method='cubic'),  # 3阶B样条曲线差值
        'bfill': missing_value_dF.fillna(method='bfill'),  # 从缺失值后面的有效值从后向前填充
        'backfill': missing_value_dF.fillna(method='bfill'),  # 从缺失值后面的有效值从后向前填充
        'ffill': missing_value_dF.fillna(method='ffill'),  # 从缺失值前面的有效值从前向后填充
        'pad': missing_value_dF.fillna(method='ffill'),  # 从缺失值前面的有效值从前向后填充
        'mean': missing_value_dF.fillna(value=missing_value_dF.mean()),  # 使用均值填补缺失值
        'mode': missing_value_dF.fillna(value=missing_value_dF.mode().mean()),  # 使用众数填补缺失值
        'median': missing_value_dF.fillna(value=missing_value_dF.median())  # 使用中位数填补缺失值
    }

    # 遍历每个特征列和对应的填补方法
    for colName, fill_Methods in colNames_fill_Methods.items():
        for fill_method in fill_Methods:
            fillCol = f"{colName}_{fill_method}"  # 填补后的特征列的列名
            fillCols.append(fillCol)
            fill_method_Df[fillCol] = fill_methods_dict[fill_method][colName]  # 执行填补操作

    fill_method_Df = pd.DataFrame(fill_method_Df)  # 将填补后的特征列转换为DataFrame对象

    # 简单填补方法拟合效果
    legend_ = ["original_data"]
    fig = plt.figure(dpi=128, figsize=(10, 20))
    for k, fillCol in enumerate(fillCols):
        ax = fig.add_subplot(round(len(fillCols) / 2), 2, k + 1)
        original_data[fillCol[:fillCol.find('_')]].plot.kde(ax=ax)
        fill_method_Df[fillCol].plot.kde(ax=ax)
        legend_.append(fillCol)
        ax.legend(legend_, loc='best')
        del legend_[-1]
    # 显示图形
    plt.show()

    return fill_method_Df, fillCols  # 返回填补后的特征列和填补后特征列的列名


# 调用缺失值填补方法对比模块
fill_method_contrast(data_df=data_missing,
                     colNames_fill_Methods={'age': ['mean', 'mode', 'median',
                                                    'bfill', 'backfill', 'ffill',
                                                    'nearest', 'zero',
                                                    'slinear', 'linear',
                                                    'quadratic', 'cubic']})

