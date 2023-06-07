import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

pima = pd.read_csv(r"E:\gitlocal\ml_code\diabetes.csv", dtype='float')
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


import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_process.missvalue_filler import CategoricalMissingValueFiller
from data_process.missvalue_filler import ContinuousMissingValueFiller, ordered_encode_columns, one_hot_encode_columns


obj = ContinuousMissingValueFiller(data_missing, fillCols=["BMI"])
# obj.kmeans_group_fill_method(corrCol="age")
obj.sequential_kmeans_group_fill_method(clusterCol="age")
