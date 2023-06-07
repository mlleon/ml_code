import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.interpolate import lagrange


def basic_fill_method_contrast_(data_df, colNames_fill_Methods):
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

    :调用方式参考:调用缺失值填补方法对比模块
    fill_method_contrast(data_df=data_missing,
                         colNames_fill_Methods={'age': ['mean', 'mode', 'median',
                                                        'bfill', 'backfill', 'ffill',
                                                        'nearest', 'zero',
                                                        'slinear', 'linear',
                                                        'quadratic', 'cubic']})
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


def basic_fill_method_(data_df, colNames_fill_Methods):
    """缺失值填补方法

        :param data_df：含有缺失值的数据集
        :param colNames_fill_Methods：字典类型，key-缺失值特征列, value-缺失值填补方法
        :fill_Method: 常值（0,1...），均值('mean')，众数（'mode'）
        :fill_Method: 从缺失值后面的有效值从后向前填充('bfill', 'backfill')
        :fill_Method: 从缺失值前面的有效值从前向后填充('ffill', 'pad')
        :fill_Method: 使用差值函数插补缺失值：
                            'nearest':最近邻插值法,
                            'zero':阶梯差值,
                            'slinear','linear':线性差值,
                            'quadratic', 'cubic'：2,3阶B样条曲线差值

        :return：返回缺失值填补后的特征列组合的DataFrame对象

        :调用方式参考：调用缺失值填补方法模块
        missing_value_dF, fillCols = fill_missing_value(data_df=data_missing,
                                                        colNames_fill_Methods=
                                                        {'age': ['ffill'],
                                                         '怀孕次数': ['ffill'],
                                                         '2小时耐量实验值': ['ffill'],
                                                         '舒张压': ['nearest'],
                                                         '三头肌皮褶厚度': ['nearest'],
                                                         '2小时血清胰岛素浓度': ['nearest'],
                                                         'BMI': ['nearest'],
                                                         '遗传函数': ['nearest']})
    """
    original_data = data_df.copy()
    missing_value_dF = data_df.copy()
    fillCols = []

    for colName, fill_Methods in colNames_fill_Methods.items():
        for fill_method in fill_Methods:
            fillCol = f"{colName}_{fill_method}"
            fillCols.append(fillCol)

            if fill_method in ['nearest', 'zero', 'slinear', 'linear', 'quadratic', 'cubic']:
                # 使用差值函数插补缺失值
                missing_value_dF[colName] = missing_value_dF[colName].interpolate(method=fill_method)
            elif fill_method in ['bfill', 'backfill', 'ffill', 'pad']:
                # 使用普通差值插补缺失值
                missing_value_dF[colName] = missing_value_dF[colName].fillna(method=fill_method)
            elif fill_method in ['mean', 'mode', 'median']:
                # 使用数学统计方法填补缺失值
                if fill_method == 'mean':
                    fill_value = missing_value_dF[colName].mean()
                elif fill_method == 'mode':
                    fill_value = missing_value_dF[colName].mode().mean()
                elif fill_method == 'median':
                    fill_value = missing_value_dF[colName].median()
                else:
                    fill_value = None
                missing_value_dF[colName] = missing_value_dF[colName].fillna(value=fill_value)
            else:
                # 使用常数值填补缺失值
                missing_value_dF[colName] = missing_value_dF[colName].fillna(value=fill_method)

    # 简单填补方法拟合效果
    legend_ = ["original_data"]
    fig = plt.figure(dpi=128, figsize=(10, 20))
    for k, fillCol in enumerate(fillCols):
        ax = fig.add_subplot(round(len(fillCols) / 2), 2, k + 1)
        original_data[fillCol[:fillCol.find('_')]].plot.kde(ax=ax)
        missing_value_dF[fillCol[:fillCol.find('_')]].plot.kde(ax=ax)
        legend_.append(fillCol)
        ax.legend(legend_, loc='best')
        del legend_[-1]
    # 显示图形
    plt.show()

    return missing_value_dF, fillCols


def kmeans_group_fill_method_(corrCol, fillCols, data_df, n_clusters, only_missValue=True):
    """K-means分层均值缺失值填补方法

        :param corrCol：与缺失值列（A）相关性最强的一个属性（B）
        :param fillCols：缺失值列（A），列表类型
        :param data_df：原始数据集
        :param n_clusters：K-means聚类算法的聚类中心数
        :param only_missValue：是否只填充缺失值NaN，默认为True
        :param only_missValue：若为False，对缺失值NaN和0值填补

        :return：返回缺失值填补后的的DataFrame对象

        :代码调用参考：调用分层均值填补缺失值函数填补异常值（K-means方法）
        data_df, cluster_count = kmeans_filling(corrCol='age', fillCols=['2小时血清胰岛素浓度', 'BMI', '遗传函数'],
                                                data_df=data_missing, n_clusters=5, only_missValue=True)
    """
    # 复制原始数据集
    data_df = data_df.copy()
    original_data = data_df.copy()

    # 使用属性B（与缺失值列A相关性最强）的众数均值填充缺失值列A的缺失值
    data_df[corrCol].fillna(data_df[corrCol].mode().mean(), inplace=True)

    # 获取训练集，Sklearn处理对象至少是二维数组，需要reshape
    x_train = data_df[corrCol].values.reshape(-1, 1)

    # 使用K-means聚类算法对数据集进行分组
    cluster_values = KMeans(n_clusters, random_state=9).fit_predict(x_train)
    data_df['cluster_values'] = cluster_values

    # 计算每个分组的数据数量
    cluster_count = data_df.groupby('cluster_values')[fillCols].count()

    # 获取包含唯一分类值的列表
    clusters_ = list(set(data_df['cluster_values']))

    for fillCol in fillCols:
        for cluster in clusters_:
            # 获取每个分组的布尔值
            ser = data_df['cluster_values'] == cluster

            # 获取缺失值列在当前分组中的数据
            fillCol_s = data_df.loc[ser, fillCol]

            # 计算当前分组中缺失值列的均值
            cluster_mean = fillCol_s.mean()

            # 获取缺失值或0值的索引列表
            if only_missValue:
                bool_index = fillCol_s.index[fillCol_s.isnull()]
            else:
                bool_index = fillCol_s.index[(fillCol_s.isnull() | (fillCol_s == 0))]

            # 使用分组均值填充缺失值或0值
            data_df.loc[bool_index, fillCol] = cluster_mean

    # 绘制原始数据和填充后数据的核密度估计图
    fig, axes = plt.subplots(round(len(fillCols) / 2), 2, figsize=(10, 10), dpi=128)
    axes = axes.ravel()
    legend_ = ["original_data"]

    for k, fillCol in enumerate(fillCols):
        original_data[fillCol].plot.kde(ax=axes[k])
        data_df[fillCol].plot.kde(ax=axes[k])
        legend_.append(fillCol)
        axes[k].legend(legend_, loc='best')
        legend_.pop()
    # 显示图形
    plt.show()

    return data_df, cluster_count


def sequential_kmeans_group_fill_method_(clusterCol, original_data_df, n_clusters, fill_method):
    """K-means分层均值缺失值填补方法

   :param clusterCol：选定的数据集分类列
   :param original_data_df：原始数据集
   :param n_clusters：K-means聚类算法的聚类中心数
   :param fill_method：分组后填补缺失值方法

   :return：返回缺失值填补后的的DataFrame对象

   :代码调用参考：调用序贯热平台填补缺失值方法
    sequential_hot_platform_fill(clusterCol='age',
                                original_data_df=data_missing,
                                n_clusters=4,fill_method='bfill')
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
        # 获取每个分组中缺失值计数情况
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

    # 3.计算相关系数,选择与缺失属性相关性最高的属性进行数据重新排序,并填补缺失值
    # 填补完毕后用填补过的属性替换分组数据集中的相应字段
    for i in range(df_LF.shape[0]):
        for j in range(df_LF.shape[1]):
            # 计算协方差矩阵
            # corrs = df_LF[i][j].corr()
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
            s.loc[:, df_null_index[j]] = df_LF[i][j][df_null_index[j]]
            cluster_full_data_df[i] = s.copy()

    # 最后拼接每组填补好的数据,并重新按照索引排序,恢复数据原本排列
    data_fill = pd.concat([i for i in cluster_full_data_df])
    data_fill = data_fill.sort_index()

    # 查看序贯热平台填补完毕后数据的缺失值情况
    nulldata1 = data_fill.isnull().sum()

    # 序贯热平台填补后对少量未补全的缺失值数据填补
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
        ax = fig.add_subplot(round(len(fillCols) / 2), 2, i + 1)
        original_data[fillCol].plot.kde(ax=ax)
        data_full_fill[fillCol].plot.kde(ax=ax)
        legend_.append(fillCol)
        ax.legend(legend_, loc='best')
        del legend_[-1]
    # 显示图形
    plt.show()

    return cluster_count, nulldata1, data_full_fill, nulldata2


def lagrange_fill_method_(data_df, fillCols):
    """拉格朗日插补缺失值方法

   :param data_df：原始数据集
   :param fillCols：进行缺失值填补的列索引名称，列表形式

   :return：返回缺失值填补后的的DataFrame对象

   :代码调用参考：调用拉格朗日插值函数,对单个或多个特征进行缺失值填补
    data_df = lg_fill_missing_value(data_df=data_missing, fillCols=['age'])
   """

    # 复制一个数据集
    data_df = data_df.copy()
    original_data = data_df.copy()

    # 2. 定义拉格朗日函数
    def lg(datasets):
        # 自定义列向量插值函数
        def ployinterp_column(s, n, k=3):
            y = s.reindex(list(range(n - k, n + 1 + k)))  # 取数
            y = y[y.notnull()]  # 剔除空值
            result = lagrange(y.index, list(y))(n)
            return result  # 插值并返回插值结果

        # 逐个元素判断是否需要插值
        for p in datasets.columns:
            for q in range(len(datasets)):
                if (datasets[p].isnull())[q]:  # 如果为空即插值
                    datasets.loc[q, p] = ployinterp_column(datasets[p], q)
        return datasets

    # 3. 调用拉格朗日插值函数填补缺失值
    # 3.1.1 调用拉格朗日插值函数,对单个或多个特征进行缺失值填补
    for fillcol in fillCols:
        fillcol_df = pd.DataFrame(np.array(data_df[fillcol]).reshape(-1, 1))
        lg(fillcol_df)
        data_df[fillcol] = fillcol_df

    # 3.1.2 拉格朗日插值函数填补结果拟合效果可视化
    legend_ = ["original_data"]
    fig = plt.figure(dpi=128, figsize=(8, 2))

    for i, fillCol in enumerate(fillCols):
        ax = fig.add_subplot(math.ceil(len(fillCols) / 2), 2, i + 1)
        original_data[fillCol].plot.kde(ax=ax)
        data_df[fillCol].plot.kde(ax=ax)
        legend_.append(fillCol)
        ax.legend(legend_, loc='best')
        del legend_[-1]
    # 显示图形
    plt.show()

    return data_df


def model_fit_fill_method_(data_missing_df, model, fillcols, fill0=False):
    """
    算法拟合填补完整列后不用于下一个特征的缺失值或异常值预测，仍使用未填补前的数据集预测

    :param data_missing_df: 原始数据集
    :param model: 算法模型（决策树、KNN、SVM、随机森林等）
    :param fillcols: 进行缺失值和异常值填补（如0值）的列索引名称，列表形式
    :param fill0: 默认为False，不对隐性缺失值填充（如0值）

    :return: 返回缺失值填补后的DataFrame对象和填补前后数据统计量变化对比值

    :代码调用参考：调用填补完整的特征列不用于下一个缺失值或异常值的算法拟合填补过程
    fill_rf_reg, statistical_contrast = fill_value_algorithm(data_missing_df=data_missing,
                                                             fillcols=['2小时耐量实验值', '舒张压',
                                                                       '三头肌皮褶厚度', '2小时血清胰岛素浓度',
                                                                       'BMI', '遗传函数', 'age'],
                                                             fill0=False,
                                                             model=RandomForestRegressor(n_estimators=100))
    """

    data_df = data_missing_df.copy()
    original_data = data_missing_df.copy()

    """1. 使用算法拟合插补缺失值NaN"""
    # 获取含有缺失值NaN的列索引
    sortindex = data_df.isnull().sum().sort_values().index

    # 依次按照列缺失值多少从少到多填补缺失值NaN
    for i in sortindex:
        # 不含有缺失值NaN的属性不参与填补
        if data_df.loc[:, i].isnull().sum() != 0:
            # 选取含有缺失值NaN属性"i"
            fillcol = data_df.loc[:, i]

            # 获取不包含缺失值NaN属性"i"的DataFrame对象
            df = data_df.loc[:, data_df.columns != i]

            # 在不包含缺失值NaN属性"i"的新特征矩阵中，对含有缺失值NaN的列，进行0的填补，为了后续使用模型
            df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

            # 划分训练集和测试集
            # 选取训练集标签值（含有缺失值属性"i"的非缺失值数据）
            Ytrain = fillcol[fillcol.notnull()]
            # 选取训练集特征值矩阵
            Xtrain = df_0[Ytrain.index, :]

            # 选取测试集标签值（含有缺失值属性"i"的缺失值数据）
            Ytest = fillcol[fillcol.isnull()]
            # 选取测试集特征值矩阵
            Xtest = df_0[Ytest.index, :]

            # 调用算法模型来填补缺失值
            estimator = model  # 实例化一个估计器
            estimated = estimator.fit(Xtrain, Ytrain)  # 模型训练
            Ypredict = estimated.predict(Xtest)  # 输出预测值

            # 将预测的缺失值放回原数据集中
            data_df.loc[Ytest.index, Ytest.name] = Ypredict

    """2. 使用算法拟合更改隐性异常0值"""
    if fill0:
        # 获取含有隐性异常0值的列索引
        sortindex = (data_df[fillcols] == 0).astype(int).sum(axis=0).sort_values().index
        # 依次按照列隐性异常0值多少从少到多更改异常0值
        for i in sortindex:
            if (data_df.loc[:, i] == 0).astype(int).sum(axis=0) != 0:
                # 选取含有0值的属性"i"（填补列）
                fillcol = data_df.loc[:, i]
                # 获取不包含0值的属性"i"的DataFrame对象
                df = data_df.loc[:, data_df.columns != i]

                # 划分训练集和测试集
                # 选取训练集标签值（含有0值属性"i"的非0值数据）
                Ytrain = fillcol[fillcol != 0]
                # 选取训练集特征值矩阵
                Xtrain = df.iloc[Ytrain.index]

                # 选取测试集标签值（含有0值属性"i"的0值数据）
                Ytest = fillcol[fillcol == 0]
                # 选取测试集特征值矩阵
                Xtest = df.iloc[Ytest.index]

                # 调用算法模型来填补缺失值
                estimator = model  # 实例化一个估计器
                estimated = estimator.fit(Xtrain, Ytrain.astype('int'))  # 模型训练，标签必须是int型
                Ypredict = estimated.predict(Xtest)  # 输出预测值

                # 将预测的缺失值放回原数据集中
                data_df.loc[Ytest.index, Ytest.name] = Ypredict

    # 统计量/固定值占位后算法拟合迭代填充后与填补前原始数据的统计量变化对比
    statistical_contrast = data_df.mean() - data_missing_df.mean()

    # 3. 算法拟合填补效果可视化
    legend_ = ["原始数据"]
    fig = plt.figure(dpi=128, figsize=(10, 15))
    fillCols = list(data_df.columns)
    for i, fillCol in enumerate(fillCols):
        ax = fig.add_subplot(math.ceil(len(fillCols) / 2), 2, i + 1)
        original_data[fillCol].plot.kde(ax=ax)
        data_df[fillCol].plot.kde(ax=ax)
        legend_.append(fillCol)
        ax.legend(legend_, loc='best')
        del legend_[-1]
    # 显示图形
    plt.show()

    # 遍历完所有属性后，返回填补好缺失值的数据集和填补前后数据统计量变化对比
    return data_df, statistical_contrast


# 填补完整的特征列用于下一个缺失值或异常值的算法拟合填补过程
def iterate_model_fit_fill_method_(data_missing_df, model, fillcols, fill0=False):
    """算法拟合填补完整列后用于下一个特征的异常值预测

   :param data_missing_df: 原始数据集
   :param model: 算法模型（决策树、KNN、SVM、随机森林等）
   :param fillcols: 进行缺失值和异常值填补（如0值）的列索引名称，列表形式
   :param fill0: 默认为False，不对隐性缺失值填充（如0值）

   :return：返回缺失值填补后的的DataFrame对象和填补前后数据统计量变化对比值

   :代码调用参考：调用填补完整的特征列用于下一个缺失值或异常值的算法拟合填补过程，使用随机森林填补缺失值
    fill_rf_reg, statistical_contrast = fill_value_algorithm_iterate(data_missing_df=data_missing,
                                                                     model=RandomForestRegressor(),
                                                                     fillcols=['2小时耐量实验值', '舒张压',
                                                                               '三头肌皮褶厚度',
                                                                               '2小时血清胰岛素浓度',
                                                                               'BMI', '遗传函数', 'age'])
   """

    data_df = data_missing_df.copy()
    original_data = data_missing_df.copy()

    """1.使用算法拟合插补缺失值NaN"""
    # 1、将完整属性和不完整属性区分开来
    # 选择按列进行缺失值统计后，统计值为0的属性名，作为完整初始完整属性
    full_feat_index = data_df.isnull().sum()[data_df.isnull().sum() == 0].index
    full_feature = data_df.loc[:, full_feat_index]

    # 选择按列进行缺失值统计后，统计值不为0的属性名，作为待填补缺失值的属性
    null_index = data_df.isnull().sum()[data_df.isnull().sum() != 0].index
    null_feature = data_df.loc[:, null_index]

    # 将不完整属性按照缺失值数量从小到大排列，并按照顺序输出其属性名称
    null_index_sorted = null_feature.isnull().sum().sort_values().index

    # 2、逐个对不完整属性进行缺失值填补（顺序为从少到多）
    for i in range(len(null_index_sorted)):
        # 2.1使用full_feature作为特征矩阵，依次使用缺失值最少的属性作为目标变量
        x, y = full_feature, data_df[null_index_sorted[i]]

        # 2.2划分训练集和测试集
        # 使用目标变量列值不为空的样本作为训练集，将所有目标变量列值为空的样本作为测试集
        y_train = y[y.notnull()]  # 筛选目标变量不为空的列值为训练集的标签值
        x_train = x[y.notnull()]  # 筛选与训练集的目标变量索引一致的full_feature值为特征值

        # y_test = y[y.isnull()]  # 筛选目标变量为空的列值为测试集的标签值
        x_test = x[y.isnull()]  # 筛选与测试集的目标变量索引一致的full_feature值为特征值

        # 2.3调用机器学习算法填补缺失值
        estimator = model
        estimated = estimator.fit(x_train, y_train)
        y_pred = estimated.predict(x_test)

        # 2.4将填补好的值放回原Dataframe中（下述几种从方式，任意—种均可)
        data_df.loc[y.isnull(), y.name] = y_pred.round(2)

        # 将填补好的属性用于下一个特征的缺失值预测
        full_feature.loc[:, null_index_sorted[i]] = data_df[null_index_sorted[i]]

    """2.使用算法拟合更改隐性异常0值"""
    if fill0:
        data_df = data_df.copy()

        # 1、将包含隐性异常值0的属性和不包含隐性异常值0的属性区分开
        not_outlier_cols = []  # 创建不包含隐性异常值0的属性索引列表，作为初始完整属性
        # 1.1获取data_df中不在fillcols中的不包含隐性异常值0的属性
        for col in data_df.columns:
            if col not in fillcols:
                not_outlier_cols.append(col)

        # 1.2 获取data_df中在fillcols中的不包含隐性异常值0的属性
        # 统计DataFrame对象每列中零值的个数
        count_0_df = (data_df == 0).astype(int).sum(axis=0)
        no_0_cols = count_0_df[count_0_df == 0].index
        for col_ in no_0_cols:
            # 将data_df中在fillcols中的不包含隐性异常值0的属性添加到not_outlier_cols列表
            not_outlier_cols.append(col_)
            # 将data_df中在fillcols中的不包含隐性异常值0的属性移除fillcols列表
            fillcols.remove(col_)

        # 选择不含有异常值的列作为待更改异常值列的初始属性
        not_outlier_feature = data_df.loc[:, not_outlier_cols]

        # 将不完整属性按照缺失值数量从小到大排列，并按照顺序输出其属性名称
        count_0_index_sorted = (data_df[fillcols] == 0).astype(int).sum(axis=0).sort_values().index

        # 2、逐个对不完整属性进行缺失值填补（顺序为从少到多）
        for i in range(len(count_0_index_sorted)):
            # 2.1使用not_outlier_feature作为特征矩阵，依次使用异常值最少的属性作为目标变量
            x, y = not_outlier_feature, data_df[count_0_index_sorted[i]]

            # 2.2划分训练集和测试集
            # 使用目标变量列值不为异常值的样本作为训练集，将所有目标变量列值为异常值的样本作为测试集
            y_train = y[y != 0]  # 筛选目标变量不为异常值的列值为训练集的标签值
            x_train = x[y != 0]  # 筛选与训练集的目标变量索引一致的not_outlier_feature值为特征值

            y_test = y[y == 0]  # 筛选目标变量为异常值的列值为测试集的标签值
            x_test = x[y == 0]  # 筛选与测试集的目标变量索引一致的not_outlier_feature值为特征值

            # 2.3调用机器学习算法更改异常值
            estimator = model
            estimated = estimator.fit(x_train, y_train)
            y_pred = estimated.predict(x_test)

            # 2.4将填补好的值放回原Dataframe中（下述几种从方式，任意—种均可)
            data_df.loc[y_test.index, y_test.name] = y_pred.round(2)
            # data_df.loc[y_test.index,y_test.name] = y_pred
            # data_df.loc[y_test.index,nulL_index_sorted[i]] = y_pred

            # 将填补好的属性用于下一个特征的缺失值预测
            not_outlier_feature.loc[:, count_0_index_sorted[i]] = data_df[count_0_index_sorted[i]]

    # 统计量/固定值占位后算法拟合迭代填充后与填补前原始数据的统计量变化对比
    statistical_contrast = data_df.mean() - data_missing_df.mean()

    # 3.算法拟合填补效果可视化
    legend_ = ["original_data"]
    fig = plt.figure(dpi=128, figsize=(10, 15))
    fillCols = list(data_df.columns)
    for i, fillCol in enumerate(fillCols):
        ax = fig.add_subplot(math.ceil(len(fillCols) / 2), 2, i + 1)
        original_data[fillCol].plot.kde(ax=ax)
        data_df[fillCol].plot.kde(ax=ax)
        legend_.append(fillCol)
        ax.legend(legend_, loc='best')
        del legend_[-1]
    # 显示图形
    plt.show()

    # 遍历完所有属性后,返回填补好缺失值的数据集和填补前后数据统计量变化对比
    return data_df, statistical_contrast