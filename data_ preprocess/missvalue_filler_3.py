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

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy.interpolate import lagrange


def extract_string(input_string):
    import re
    """
    从字符串中提取字符串部分
    参数:
    - input_string: 输入字符串
    返回值:
    - extracted_string: 提取的字符串部分
    """
    input_string = str(input_string)
    pattern = r'^([A-Za-z]+)'
    match = re.match(pattern, input_string)
    if match:
        return match.group(1)


# 使用有序数字编码方式对分类特征编码（考虑数值大小意义）
def ordered_encode_columns(ori_df, delCols=None, fillCols=None, outlier=None):
    data_df = ori_df.copy()

    if fillCols is None:
        if delCols is None:
            fillCols = data_df.columns
        else:
            fillCols = [col for col in data_df.columns if col not in delCols]

    col_dtype = {}
    for col in fillCols:
        data_type = data_df[col].dtype
        if extract_string(data_type) == "int":
            col_dtype[col] = "int"
            min_value = data_df[col].min()
            max_value = data_df[col].max()
            if np.isinf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.inf, data_df[col].replace(np.inf, min_value).max().max())
            if np.isneginf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.NINF, data_df[col].replace(np.NINF, max_value).min().min())
            if data_df[col].isnull().sum():
                data_df[col] = data_df[col].fillna(round(min_value - 1, 0))

            if outlier is not None:
                if col in [col_ for col_ in outlier.keys()]:
                    data_df[col] = data_df[col].replace(outlier[col], round(min_value - 1, 0))
            else:
                continue
        elif extract_string(data_type) == "float":
            col_dtype[col] = "float"
            min_value = data_df[col].min()
            max_value = data_df[col].max()
            if np.isinf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.inf, data_df[col].replace(np.inf, min_value).max())
            if np.isneginf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.NINF, data_df[col].replace(np.NINF, max_value).min())
            if data_df[col].isnull().sum():
                data_df[col] = data_df[col].fillna(round(min_value - 1, 0))
            if outlier is not None:
                if col in [col_ for col_ in outlier.keys()]:
                    data_df[col] = data_df[col].replace(outlier[col], round(min_value - 1, 0))
            else:
                continue
        elif extract_string(data_type) == "object":
            col_dtype[col] = "object"

            if outlier is not None:
                if col in [col_ for col_ in outlier.keys()]:
                    data_df[col] = data_df[col].replace(outlier[col], -1)

            data_df_ = data_df.copy()

            data_df[col] = data_df[col].fillna("ZZZZZZZZZZZZZZZZZ")
            value = data_df[col].unique().tolist()
            value.sort()
            series = pd.Series(data_df[col].map(pd.Series(range(len(value)), index=value)).values)

            if data_df_[col].isnull().sum():
                max_value = series.max()
                data_df[col] = series.replace(max_value, -1)
            else:
                data_df[col] = series

        elif extract_string(data_type) == "datetime":
            col_dtype[col] = "datetime"
            min_value = data_df[col].min()
            max_value = data_df[col].max()
            if np.isinf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.inf, data_df[col].replace(np.inf, min_value).max().max())
            if np.isneginf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.NINF, data_df[col].replace(np.NINF, max_value).min().min())
            if data_df[col].isnull().sum():
                data_df[col] = data_df[col].fillna(min_value - pd.Timedelta(days=1))
            else:
                continue
        elif extract_string(data_type) == "timedelta":
            col_dtype[col] = "timedelta"
            min_value = data_df[col].min()
            max_value = data_df[col].max()
            if np.isinf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.inf, data_df[col].replace(np.inf, min_value).max().max())
            if np.isneginf(data_df[col]).any():
                data_df[col] = data_df[col].replace(np.NINF, data_df[col].replace(np.NINF, max_value).min().min())
            if data_df[col].isnull().sum():
                data_df[col] = data_df[col].fillna(min_value - pd.Timedelta(days=1))
            else:
                continue
        else:
            print(f"没有定义该数据类型: {data_type}")

    return data_df, col_dtype


# 将每列最小值转化为NAN
def fill_miss_value(ori_df, encode_df, fillCols, fill_method, model, n_clusters, fill_mode, draw):
    encode_df_ = encode_df.copy()
    for col in fillCols:
        if ori_df[col].isnull().sum():
            # 查找最小值
            min_value = encode_df_[col].min()
            # 使用 replace 方法将最小值替换为NAN
            encode_df_[col] = encode_df_[col].replace(min_value, np.nan)

            # 填补缺失值NAN, 实例化缺失值填补类
            obj_filler = MissingValueFiller(data_df=encode_df_, fillCols=[col])

            if fill_mode == "contrast":
                encode_df_ = obj_filler.fill_method_contrast(fill_methods=fill_method, draw=draw)
            elif fill_mode == "basic":
                encode_df_ = obj_filler.basic_fill_method(fill_methods=fill_method, draw=draw)
            elif fill_mode == "kmeans":
                encode_df_ = obj_filler.kmeans_group_fill_method(
                    fillMethod=fill_method, n_clusters=n_clusters, only_missValue=True, draw=draw)
            elif fill_mode == "sequential":
                encode_df_ = obj_filler.sequential_kmeans_group_fill_method(
                    fill_method=fill_method, n_clusters=n_clusters, draw=draw)
            elif fill_mode == "lagrange":
                encode_df_ = obj_filler.lagrange_fill_method(draw=False)
            elif fill_mode == "model":
                encode_df_ = obj_filler.model_fit_fill_method(model=model, fill0=False, draw=draw)

    return encode_df_, fillCols


class MissingValueFiller:
    def __init__(self, data_df, fillCols=None):
        self.original_data = data_df.copy()
        self.missing_value_dF = data_df.copy()
        self.fillCols = fillCols

    # 某个特征常规缺失值填补方法对比, 最好一次对比一个特征
    def fill_method_contrast(self, fillCols=None, fill_methods=None, draw=False):
        """
        :fillCols 列表，填补缺失值列名称
        :fill_methods: 列表（缺失值填补方法）
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
        original_data = self.original_data
        missing_value_dF = self.missing_value_dF

        if fillCols is None:
            fillCols = self.fillCols

        if isinstance(fillCols, str):
            fillCols = [fillCols]

        if isinstance(fill_methods, str):
            fill_methods = [fill_methods]

        fill_method_Df = {}  # 用于存储填补后的df
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
        fillCols_ = []

        assert len(fillCols) == 1, "The column names more than one, you can only enter one column name."

        colNames_fill_Methods = {col: fill_methods for col in fillCols}
        for colName, fill_Methods in colNames_fill_Methods.items():
            for fill_method in fill_Methods:
                fillCol = f"{colName}_{fill_method}"  # 填补后的特征列的列名
                fillCols_.append(fillCol)
                fill_method_Df[fillCol] = fill_methods_dict[fill_method][colName]  # 执行填补操作

        fill_method_Df = pd.DataFrame(fill_method_Df)  # 将填补后的特征列转换为DataFrame对象

        if draw:
            # 简单填补方法拟合效果
            legend_ = ["original_data"]
            fig = plt.figure(dpi=128, figsize=(10, 20))
            for k, fillCol in enumerate(fillCols_):
                ax = fig.add_subplot(math.ceil(len(fillCols_) / 2), 2, k + 1)
                original_data[fillCol[:fillCol.find('_')]].plot.kde(ax=ax)
                fill_method_Df[fillCol].plot.kde(ax=ax)
                legend_.append(fillCol)
                ax.legend(legend_, loc='best')
                del legend_[-1]
            # 显示图形
            plt.show()

        return fill_method_Df  # 返回填补后的特征列和填补后特征列的列名

    # 适用于任何分类和连续型数据类型
    def basic_fill_method(self, fillCols=None, fill_methods=None, draw=False):
        """常规缺失值填补方法
            :fillCols：列表类型，缺失值填补列名称

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
            missing_value_dF, fillCols = basic_fill_method(data_df=data_missing,
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
        original_data = self.original_data
        missing_value_dF = self.missing_value_dF
        if fillCols is None:
            fillCols = self.fillCols

        if isinstance(fillCols, str):
            fillCols = [fillCols]

        if isinstance(fill_methods, str):
            fill_methods = [fill_methods]

        fillCols_ = []  # 存储填补后的特征列的列名

        assert len(fillCols) == len(
            fill_methods), "The column name list and corresponding method list do not have the same length"
        colNames_fill_Methods = {col: [meth] for col, meth in zip(fillCols, fill_methods)}
        for colName, fill_Methods in colNames_fill_Methods.items():
            for fill_method in fill_Methods:
                fillCol = f"{colName}_{fill_method}"
                fillCols_.append(fillCol)

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

        if draw:
            # 简单填补方法拟合效果
            legend_ = ["original_data"]
            fig = plt.figure(dpi=128, figsize=(10, 20))
            for k, fillCol in enumerate(fillCols_):
                ax = fig.add_subplot(math.ceil(len(fillCols_) / 2), 2, k + 1)
                original_data[fillCol[:fillCol.find('_')]].plot.kde(ax=ax)
                missing_value_dF[fillCol[:fillCol.find('_')]].plot.kde(ax=ax)
                legend_.append(fillCol)
                ax.legend(legend_, loc='best')
                del legend_[-1]
            # 显示图形
            plt.show()

        return missing_value_dF

    # 如果有相关性比较强的列，使用该方法的效果较好，适用于任何分类和连续型数据类型
    def kmeans_group_fill_method(self, fillCols=None,
                                 fillMethod="ffill", n_clusters=2, only_missValue=True, draw=False):
        """K-means分层均值缺失值填补方法

            :param draw:
            :param fillMethod:
            :param fillCols:
            :param n_clusters：K-means聚类算法的聚类中心数
            :param only_missValue：是否只填充缺失值NaN，默认为True
            :param only_missValue：若为False，对缺失值NaN和0值填补

            :return：返回缺失值填补后的的DataFrame对象

            :代码调用参考：调用分层均值填补缺失值函数填补异常值（K-means方法）
            data_df, cluster_count = kmeans_filling(corrCol='age', fillCols=['2小时血清胰岛素浓度', 'BMI', '遗传函数'],
                                                    data_df=data_missing, n_clusters=5, only_missValue=True)
        """

        data_df = self.missing_value_dF
        original_data = self.original_data
        if fillCols is None:
            fillCols = self.fillCols

        if isinstance(fillMethod, list):
            fillMethod = fillMethod[0]

        for fillCol in fillCols:
            # 计算所有列与列A的相关性系数
            correlation = original_data.corr()[fillCol].abs()
            # 找到与列A相关性最强的另一列B
            corrCol = correlation.drop(fillCol).idxmax()
            # 使用属性B（与缺失值列A相关性最强）的众数均值填充缺失值列A的缺失值
            data_df[corrCol].fillna(data_df[corrCol].mode()[0], inplace=True)
            # 获取训练集，Sklearn处理对象至少是二维数组，需要reshape
            x_train = data_df[corrCol].values.reshape(-1, 1)
            # 使用K-means聚类算法对数据集进行分组
            cluster_values = KMeans(n_clusters, random_state=9).fit_predict(x_train)
            data_df['cluster_values'] = cluster_values
            # 计算每个分组的数据数量
            cluster_count = data_df.groupby('cluster_values')[fillCols].count()
            # 获取包含唯一分类值的列表
            clusters_ = list(set(data_df['cluster_values']))
            for cluster in clusters_:
                # 获取每个分组的布尔值
                ser = data_df['cluster_values'] == cluster

                # 获取缺失值列在当前分组中的数据
                fillCol_s = data_df.loc[ser, fillCol]

                if fillCol_s.isnull().sum():
                    if fillMethod in ['nearest', 'zero', 'slinear', 'linear', 'quadratic', 'cubic']:
                        # 使用差值函数插补缺失值
                        data_df.loc[ser, fillCol] = data_df.loc[ser, fillCol].interpolate(method=fillMethod)
                        count = data_df.loc[ser, fillCol].isnull().sum()
                        if count:
                            fill_value = data_df.loc[ser, fillCol].mode()[0]
                            data_df.loc[ser, fillCol] = data_df.loc[ser, fillCol].fillna(value=fill_value)
                    elif fillMethod in ['bfill', 'backfill', 'ffill', 'pad']:
                        # 使用普通差值插补缺失值
                        data_df.loc[ser, fillCol] = data_df.loc[ser, fillCol].fillna(method=fillMethod)
                        count = data_df.loc[ser, fillCol].isnull().sum()
                        if count:
                            fill_value = data_df.loc[ser, fillCol].mode()[0]
                            data_df.loc[ser, fillCol] = data_df.loc[ser, fillCol].fillna(value=fill_value)
                    elif fillMethod in ['mean', 'mode', 'median']:
                        # 使用数学统计方法填补缺失值
                        if fillMethod == 'mean':
                            fill_value = data_df.loc[ser, fillCol].mean()
                        elif fillMethod == 'mode':
                            fill_value = data_df.loc[ser, fillCol].mode()[0]
                        elif fillMethod == 'median':
                            fill_value = data_df.loc[ser, fillCol].median()
                        else:
                            fill_value = np.nan
                        data_df.loc[ser, fillCol] = data_df.loc[ser, fillCol].fillna(value=fill_value)
                    else:
                        # 使用常数值填补缺失值
                        data_df.loc[ser, fillCol] = data_df.loc[ser, fillCol].fillna(value=fillMethod)

                count = data_df[fillCol].isnull().sum()

        if draw:
            # 绘制原始数据和填充后数据的核密度估计图
            fig, axes = plt.subplots(math.ceil(len(fillCols) / 2), 2, figsize=(10, 10), dpi=128)
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

        return data_df

    def sequential_kmeans_group_fill_method(self, fillCols=None, fill_method="ffill", n_clusters=2, draw=False):
        """K-means分层均值缺失值填补方法

       :param draw:
       :param fillCols:
       :param n_clusters：K-means聚类算法的聚类中心数
       :param fill_method：分组后填补缺失值方法

       :return：返回缺失值填补后的的DataFrame对象

       :代码调用参考：调用序贯热平台填补缺失值方法
        sequential_hot_platform_fill(clusterCol='age',
                                    original_data_df=data_missing,
                                    n_clusters=4,fill_method='bfill')
       """
        # 1.选定某列对数据集分组或分箱
        data_df = self.missing_value_dF
        original_data = self.original_data

        if fillCols is None:
            fillCols = self.fillCols

        if isinstance(fill_method, list):
            fill_method = fill_method[0]

        # 计算所有列与列A的相关性系数
        for k, fillCol in enumerate(fillCols):
            correlation = original_data.corr()[fillCol].abs()
            # 找到与列A相关性最强的另一列B
            clusterCol = correlation.drop(fillCol).idxmax()

            # 选择某列对数据集进行分箱或者分组
            data_df[clusterCol].fillna(data_df[clusterCol].mode()[0], inplace=True)
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
            # 将分组后的DataFrame对象保存为列表（包含所有的无缺失值列和某列缺失值）
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
                # df_null_index = df[df[0] != 0].index
                df_null_index = pd.Index([fillCol])
                # 分类后每组只包含没有缺失值列的DataFrame对象
                full_feature = cluster_full_data_df[i][df_not_null_index]

                for j in range(len(df_null_index)):
                    # 复制分类后每组只包含没有缺失值列的DataFrame对象
                    C_full_feature = full_feature.copy()
                    # 将缺失值列添加到分类后每组只包含没有缺失值列的DataFrame对象后面(非累加添加，只包含单列缺失值-即一列一列添加)
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
                    test = df_LF[i][j]
                    corrs = df_LF[i][j].corr().abs().round(2)
                    # 对协方差矩阵排序，选出与缺失值列相似性最高的特征
                    best_feat_sort = corrs.sort_values(by=df_null_index[j], ascending=False)
                    best_feat = best_feat_sort.index[1]
                    # 以相似性最高的属性对df_LF对象降序排序
                    df_LF[i][j] = df_LF[i][j].sort_values(by=best_feat, ascending=False)
                    # 将某个值赋值给df_LF对象中包含缺失值的列
                    if df_LF[i][j][df_null_index[j]].isnull().sum():
                        if fill_method in ['bfill', 'backfill', 'ffill', 'pad']:
                            # 使用普通差值插补缺失值
                            df_LF[i][j][df_null_index[j]] = df_LF[i][j][df_null_index[j]].fillna(method=fill_method)
                            count = df_LF[i][j][df_null_index[j]].isnull().sum()
                            if count:
                                fill_value = df_LF[i][j][df_null_index[j]].mode()[0]
                                df_LF[i][j][df_null_index[j]] = df_LF[i][j][df_null_index[j]].fillna(value=fill_value)

                    # df_LF[i][j][df_null_index[j]] = df_LF[i][j][df_null_index[j]].fillna(method=fill_method)
                    # 对df_LF对象按照索引排序
                    df_LF[i][j] = df_LF[i][j].sort_index()
                    # 复制cluster_full_data_df对象
                    s = cluster_full_data_df[i].copy()
                    # 将df_LF对象含有缺失值列赋值给cluster_full_data_df对象对应的缺失值列
                    # df_check = df_LF[i][j]
                    s.loc[:, df_null_index[j]] = df_LF[i][j][df_null_index[j]]
                    cluster_full_data_df[i] = s.copy()

            # 最后拼接每组填补好的数据,并重新按照索引排序,恢复数据原本排列
            fill_df = pd.concat([i for i in cluster_full_data_df]).sort_index()
            data_df[fillCol] = fill_df[fillCol]

            if draw:
                # 4. 序贯热平台填补结果拟合效果可视化
                legend_ = ["original_data"]
                fig = plt.figure(dpi=128, figsize=(10, 10))

                ax = fig.add_subplot(math.ceil(len(fillCols) / 2), 2, k + 1)
                original_data[fillCol].plot.kde(ax=ax)
                data_df[fillCol].plot.kde(ax=ax)
                legend_.append(fillCol)
                ax.legend(legend_, loc='best')
                del legend_[-1]
                # 显示图形
                plt.show()

        return data_df

    # 适合连续性数据填补缺失值
    def lagrange_fill_method(self, fillCols=None, draw=False):
        """拉格朗日插补缺失值方法

       :return：返回缺失值填补后的的DataFrame对象

       :代码调用参考：调用拉格朗日插值函数,对单个或多个特征进行缺失值填补
        data_df = lg_fill_missing_value(data_df=data_missing, fillCols=['age'])
       """

        # 复制一个数据集
        data_df = self.missing_value_dF
        original_data = self.original_data
        if fillCols is None:
            fillCols = self.fillCols

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
            fillcol_df = lg(fillcol_df)
            data_df[fillcol] = fillcol_df
        if draw:
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

    def model_fit_fill_method(self, model, fillcols=None, fill0=False, draw=False):
        """
        算法拟合填补完整列后不用于下一个特征的缺失值或异常值预测，仍使用未填补前的数据集预测

        :param fillcols:
        :param model: 算法模型（决策树、KNN、SVM、随机森林等）
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

        data_df = self.missing_value_dF
        original_data = self.original_data
        if fillcols is None:
            fillcols = self.fillCols

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
        statistical_contrast = data_df.mean() - self.original_data.mean()

        if draw:
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
        return data_df

    # 填补完整的特征列用于下一个缺失值或异常值的算法拟合填补过程
    def iterate_model_fit_fill_method(self, model, fillcols=None, fill0=False, draw=False):
        """算法拟合填补完整列后用于下一个特征的异常值预测

       :param fillcols:
       :param model: 算法模型（决策树、KNN、SVM、随机森林等）
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
        data_df = self.missing_value_dF
        original_data = self.original_data
        if fillcols is None:
            fillcols = self.fillCols

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
        statistical_contrast = data_df.mean() - self.original_data.mean()

        if draw:
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
        return data_df


def miss_value_filler(ori_df,  # 最原始的DataF数据
                      delCols,  # 不需要数据编码的列，列表类型
                      fillCols,  # 需要填补缺失值的列，列表类型
                      outlier,  # 异常值字典，字典类型{col:outlier}
                      fill_method,  # 填充方法，列表类型或字符串类型
                      model,  # 填补缺失值拟合模型
                      fill_mode,  # 填充方法名称（选择哪种填充方式），字符串类型
                      n_clusters,  # K-means聚类算法的聚类中心数
                      draw):  # 是否绘制对比图
    # for col in fillCols:
    # count = ori_df[col].isnull().sum()
    # if count:
    # 对DataFrame数据进行编码，全部转为数字类型
    encode_df, col_dtype = ordered_encode_columns(ori_df=ori_df, delCols=delCols, outlier=outlier)
    # 将每列最小值转化为NAN，然后填补该列缺失值
    filled_df, fillCols = fill_miss_value(ori_df=ori_df, encode_df=encode_df, fillCols=fillCols,
                                          fill_method=fill_method, model=model, n_clusters=n_clusters,
                                          fill_mode=fill_mode, draw=draw)

    return filled_df


# # 调用某种缺失值填充方法
# ori_df = pd.read_csv(r'E:\gitlocal\ml_code\ori_dataset\merchants.csv', header=0)
# fillCols = ['category_2']
#
# filled_df = miss_value_filler(ori_df,  # 最原始的DataF数据
#                               delCols=None,  # 不需要数据编码的列，列表类型
#                               fillCols=fillCols,  # 需要填补缺失值的列，列表类型
#                               outlier=None,  # 异常值字典，字典类型{col:outlier}
#                               fill_method="bfill",  # 填充方法，列表类型或字符串类型
#                               model=RandomForestRegressor(n_estimators=100),  # 填补缺失值拟合模型
#                               fill_mode="sequential",  # 填充方法名称（选择哪种填充方式），字符串类型
#                               n_clusters=2,  # K-means聚类算法的聚类中心数
#                               draw=False)  # 是否绘制对比图
