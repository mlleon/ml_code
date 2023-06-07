import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

# 设置显示DataFrame对象的所有行和列
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('max_colwidth', 100)  # value显示宽度100，默认为50

names = ['怀孕次数', '2小时耐量实验值', '舒张压',
         '三头肌皮褶厚度', '2小时血清胰岛素浓度',
         'BMI', '遗传函数', 'age', '是否患病']

pima = pd.read_csv(r"E:\gitlocal\ml_code\diabetes.csv", dtype='float')
pima.columns = names

# 1.给原始数据随机设置缺失值
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


# -------------------------------------------------------------#

def sequential_hot_platform_fill_missing_values(clusterCol, original_data_df, n_clusters, fill_method):
    """K-means分层均值缺失值填补方法

   :param clusterCol：选定的数据集分类列
   :param data_df：原始数据集
   :param n_clusters：K-means聚类算法的聚类中心数
   :param fill_method：分组后填补缺失值方法

   :return：返回缺失值填补后的的DataFrame对象
   """
    # 1.选定某列对数据集分组或分箱

    data_df = original_data_df.copy()
    original_data = original_data_df.copy()
    # 选择某列对数据集进行分箱或者分组
    data_df[clusterCol].fillna(data_df[clusterCol].mode().mean(), inplace=True)
    # 构建分组训练集
    x_train = data_df[clusterCol].values.reshape(-1, 1)
    cluster_values = KMeans(n_clusters, random_state=9).fit_predict(x_train)
    # 将'cluster_values'列添加到data_df对象中
    data_df['cluster_values'] = cluster_values
    # 计算分组后corrCol='age'每个簇的数据数量
    cluster_count = data_df.groupby('cluster_values')[[clusterCol]].count()

    # --------------------------------------------------------------------#

    # 2.分组并保存子集
    # 将分组后的DataFrame对象保存为列表（包含所有的有缺失值列和无缺失值列）
    cluster_full_data_df = []

    # 将分组后的DataFrame对象保存为列表（只包含没有缺失值列和某列缺失值）
    LF = []

    for i in range(cluster_count.shape[0]):
        # 布尔索引获取分类后每组的DataFrame对象（包含有缺失值列和无缺失值列）
        cluster_full_data_df_subset = data_df[data_df["cluster_values"] == cluster_count.index[i]]
        # 将分类后每组的DataFrame对象添加到列表
        cluster_full_data_df.append(cluster_full_data_df_subset)
        df = pd.DataFrame(cluster_full_data_df[i].isnull().sum())
        # 布尔索引获取分类后每组的DataFrame对象没有缺失值的列索引
        df_not_null_index = df[df[0] == 0].index
        # 布尔索引获取分类后每组的DataFrame对象有缺失值的列索引
        df_null_index = df[df[0] != 0].index
        # 分类后每组只包含没有缺失值列的DataFrame对象
        full_feature = cluster_full_data_df[i][df_not_null_index]

        for j in range(len(df_null_index)):
            # 复制分类后每组只包含没有缺失值列的DataFrame对象
            C_full_feature = full_feature.copy()
            # 将缺失值列添加到分类后每组只包含没有缺失值列的DataFrame对象后面(非累加添加，只包含单列缺失值)
            C_full_feature[df_null_index[j]] = cluster_full_data_df[i][df_null_index[j]]
            # 将C_full_feature添加到LF列表
            LF.append(C_full_feature)

    # 将LF列表转化为一维array对象
    df_LF = np.array(LF, dtype=object)
    # 一维array对象转化为二维array对象
    df_LF = df_LF.reshape(len(cluster_full_data_df), -1)

    # --------------------------------------------------------------------#

    # 3.计算相关系数,选择与缺失属性相关性最高的属性进行数据重新排序,并填补缺失值
    # 填补完毕后用填补过的属性替换分组数据集中的相应字段
    for i in range(df_LF.shape[0]):
        for j in range(df_LF.shape[1]):
            # 计算协方差矩阵
            corrs = df_LF[i][j].corr()
            # 对协方差矩阵排序，选出相似性最高的特征属性
            best_feat = df_LF[i][j].corr().abs().round(2).sort_values(by=df_null_index[j]).index[-3]
            # 以相似性最高的属性对df_LF对象降序排序
            df_LF[i][j] = df_LF[i][j].sort_values(by=best_feat, ascending=False)
            # 将某个值赋值给df_LF对象中包含缺失值的列
            df_LF[i][j][df_null_index[j]] = df_LF[i][j][df_null_index[j]].fillna(method=fill_method)
            # 对df_LF对象按照索引排序
            df_LF[i][j] = df_LF[i][j].sort_index()
            # 复制cluster_full_data_df对象
            s = cluster_full_data_df[i].copy()
            # 将df_LF对象含有额缺失值列赋值给cluster_full_data_df对象对应的缺失值列
            df_check = df_LF[i][j]
            s.loc[:, df_null_index[j]] = df_LF[i][j][df_null_index[j]]
            cluster_full_data_df[i] = s.copy()

    # 最后拼接每组填补好的数据,并重新按照索引排序,恢复数据原本排列
    data_fill = pd.concat([i for i in cluster_full_data_df])
    data_fill = data_fill.sort_index()

    # 查看序贯热平台填补完毕后数据的缺失值情况
    nulldata1 = data_fill.isnull().sum()

    # 序贯热平台填补后对少量缺失值数据填补
    data_full_fill = data_fill.copy()
    for missCol in list(nulldata1[nulldata1.values != 0].index):
        data_full_fill[missCol].fillna(data_full_fill[missCol].mode().mean(), inplace=True)

    # 查看最终填补完毕后的数据缺失情况
    nulldata2 = data_full_fill.isnull().sum()

    # 4. 序贯热平台填补结果拟合效果可视化
    legend_ = ["original_data"]
    fig = plt.figure(dpi=128, figsize=(10, 10))

    fillCols = list(data_full_fill.columns)
    fillCols.remove(clusterCol)
    del fillCols[-1]

    for i, fillCol in enumerate(fillCols):
        axes = fig.add_subplot(round(len(fillCols) / 2), 2, i + 1)
        axes = original_data[fillCol].plot.kde()
        axes = data_full_fill[fillCol].plot.kde()
        legend_.append(fillCol)
        axes = axes.legend(legend_, loc='best')
        del legend_[-1]

    return cluster_count, nulldata1, data_full_fill, nulldata2


# ---------------------------------------------------------------#

# 调用序贯热平台填补缺失值方法
data_full_fill = sequential_hot_platform_fill_missing_values(clusterCol='age',
                                                             original_data_df=data_missing, n_clusters=4,
                                                             fill_method='bfill')[2]