import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from data_process.missvalue_filler import *
from sklearn import preprocessing
import numpy as np
import itertools


# def OrderedClassify(df):
# def DiscreteClassify(df):
# def ContinuousNumeric(df):


def process_timeseries(df):
    from datetime import datetime
    # 提取日期中的月份
    df['purchase_month'] = df['purchase_date'].apply(lambda x: '-'.join(x.split('-')[:2]))

    # 提取日期中的时间段（每6小时为一个时间段），得到时间段的整数表示
    df['purchase_hour_section'] = df['purchase_date'].apply(lambda x: int(x.split(' ')[1].split(':')[0]) // 6)

    # 提取日期中的星期数（工作日与周末,0表示星期一，6表示星期日）
    df['purchase_day'] = df['purchase_date'].apply(
        lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d").weekday() // 5)

    # Removing the original purchase_date column
    del df['purchase_date']

    return df


# 使用独热编码数字编码方式对分类特征编码（特征属性相互独立，即不能同时发生）
def onehot_encode_feature(data_df, fillCols=None):
    for col in fillCols:
        encoded_cols = pd.get_dummies(data_df[col], prefix=col)
        data_df = pd.concat([data_df, encoded_cols], axis=1)
        data_df.drop(col, axis=1, inplace=True)
    return data_df


def single_variate_generate_features(df, fillCols, methods, threshold=30, n_bins=5, degree=5):
    df_data = df.copy()
    for fillCol in fillCols:
        for method in methods:
            if method == "binarizer":
                # 将数据二值化
                X = df_data[fillCol].values.reshape(-1, 1)
                binarizer_ndy = Binarizer(threshold=threshold).fit_transform(X)
                # 将数组转换为Series
                series = pd.Series(binarizer_ndy.flatten())
                col_name = f"{fillCol}_{method}"
                df_data[col_name] = series

            elif method == "kbins":
                # 将数据二值化
                X = df_data[fillCol].values.reshape(-1, 1)
                est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
                kbins_ndy = est.fit_transform(X)
                # 将数组转换为Series
                series = pd.Series(kbins_ndy.flatten())
                col_name = f"{fillCol}_{method}_{n_bins}"
                df_data[col_name] = series

            elif method == "single_variable_poly":
                poly_ndy = PolynomialFeatures(degree=degree).fit_transform(df_data[fillCol].values.reshape(-1, 1))
                poly_df = pd.DataFrame(poly_ndy)
                for i, col_ in enumerate(poly_df.columns):
                    if i != 1 and i != 0:
                        df_data[f"{fillCol}_{method}_{i}"] = poly_df[col_]

    return df_data


def double_variate_cross_generate_math_features(df, fillCols, methods):
    df_data = df.copy()

    if fillCols is None:
        fillCols = df_data.columns

    for col_index, fillCol in enumerate(fillCols):
        for col_sub_index in range(col_index + 1, len(fillCols)):
            fillCol_ = fillCols[col_sub_index]
            if "add" in methods:
                df_data[f"{fillCol}_{fillCol_}_add"] = df_data[fillCol] + df_data[fillCol_]
                new_col_name = f"{fillCol}_{fillCol_}_add"
                min_value = df_data[new_col_name].min()
                max_value = df_data[new_col_name].max()
                if np.isinf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.inf, df_data[new_col_name].replace(np.inf,
                                                                                                                min_value).max())
                if np.isneginf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.NINF,
                                                                          df_data[new_col_name].replace(np.NINF,
                                                                                                        max_value).min())
            if "sub" in methods:
                new_col_name = f"{fillCol}_{fillCol_}_sub"
                df_data[new_col_name] = df_data[fillCol] - df_data[fillCol_]
                min_value = df_data[new_col_name].min()
                max_value = df_data[new_col_name].max()
                if np.isinf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.inf, df_data[new_col_name].replace(np.inf,
                                                                                                                min_value).max())
                if np.isneginf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.NINF,
                                                                          df_data[new_col_name].replace(np.NINF,
                                                                                                        max_value).min())
            if "div" in methods:
                new_col_name = f"{fillCol}_{fillCol_}_div"
                df_data[new_col_name] = df_data[fillCol] / df_data[fillCol_]
                min_value = df_data[new_col_name].min()
                max_value = df_data[new_col_name].max()
                if np.isinf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.inf, df_data[new_col_name].replace(np.inf,
                                                                                                                min_value).max())
                if np.isneginf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.NINF,
                                                                          df_data[new_col_name].replace(np.NINF,
                                                                                                        max_value).min())
                if df_data[new_col_name].isnull().sum():
                    df_data[new_col_name] = df_data[new_col_name].fillna(0)

            if "multi" in methods:
                new_col_name = f"{fillCol}_{fillCol_}_multi"
                df_data[new_col_name] = df_data[fillCol] * df_data[fillCol_]
                min_value = df_data[new_col_name].min()
                max_value = df_data[new_col_name].max()
                if np.isinf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.inf, df_data[new_col_name].replace(np.inf,
                                                                                                                min_value).max())
                if np.isneginf(df_data[new_col_name]).any():
                    df_data[new_col_name] = df_data[new_col_name].replace(np.NINF,
                                                                          df_data[new_col_name].replace(np.NINF,
                                                                                                        max_value).min())
    return df_data


def double_variate_cross_generate_features(colNames, df, OneHot=True):
    """分类变量两两组合交叉衍生函数

    :param colNames:参与交叉衍生的列名称
    :param df：原始数据集
    :param OneHot：是否进行独热编码

    :return：交叉衍生后的新特征和新列索引名称
    """
    data_df = df.copy()

    # 创建空列表存储器
    colNames_new_l = []
    features_new_l = []

    # 提取需要进行交叉组合的特征
    features = data_df[colNames]

    # 逐个创建新特征名称、新特征
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index + 1, len(colNames)):
            newNames = col_name + '&' + colNames[col_sub_index]
            colNames_new_l.append(newNames)
            newDF = pd.Series(features[col_name].astype('str')
                              + '&'
                              + features[colNames[col_sub_index]].astype('str'),
                              name=col_name)
            features_new_l.append(newDF)

    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = colNames_new_l
    colNames_new = colNames_new_l

    # 对新特征矩阵进行独热编码
    if OneHot:
        features_new = pd.get_dummies(features_new, prefix=colNames_new)

    features_new = pd.concat([data_df, features_new], axis=1)
    return features_new


def multi_variate_cross_generate_features(colNames, df, OneHot=True):
    """多变量交叉组合衍生函数

    :param colNames: 参与交叉衍生的列名称列表
    :param df: 原始数据集
    :param OneHot: 是否进行独热编码

    :return: 交叉衍生后的新特征和新列索引名称
    """
    data_df = df.copy()
    # 创建空列表存储器
    colNames_new_l = []
    features_new_l = []

    # 获取特征值列表
    feature_values = [data_df[col].unique() for col in colNames]

    # 生成所有特征组合
    feature_combinations = list(itertools.product(*feature_values))

    # 逐个创建新特征名称、新特征
    for combination in feature_combinations:
        newNames = '&'.join([col + '=' + str(val) for col, val in zip(colNames, combination)])
        colNames_new_l.append(newNames)
        newDF = pd.Series(np.where((data_df[colNames] == list(combination)).all(axis=1), 1, 0), name=newNames)
        features_new_l.append(newDF)

    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)

    # 对新特征矩阵进行独热编码
    if OneHot:
        features_new = pd.get_dummies(features_new, prefix_sep='')

    # 将新特征矩阵与原始数据集合并
    features_new = pd.concat([data_df, features_new], axis=1)
    colNames_new = features_new.columns.tolist()[len(data_df.columns):]  # 获取新列索引名称
    features_new.columns = list(data_df.columns) + colNames_new  # 更新特征矩阵的列索引名称
    return features_new


def double_variate_cross_generate_poly_features(df, fillCols, degree):
    data_df = df.copy()
    arrays = []  # 创建一个空列表
    column_names_ploy = []  # 给生成的双变量多阶多项式衍生特征生成列索引
    for col_index, fillCol in enumerate(fillCols):
        for col_sub_index in range(col_index + 1, len(fillCols)):
            double_data_df = pd.concat([data_df[fillCol], data_df[fillCols[col_sub_index]]], axis=1)
            poly_features = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            ploy_variable = poly_features.fit_transform(double_data_df)[:, 2:]
            column_temps = poly_features.get_feature_names_out(input_features=double_data_df.columns)[2:]
            for column_temp in column_temps:
                column_names_ploy.append(column_temp)
            arrays.append(ploy_variable)  # 将数组对象添加到列表中

    # 沿第1轴拼接数组对象
    concatenated_array = np.concatenate(arrays, axis=1)

    # 生成双变量多阶多项式衍升特征转换为DataFrame对象，并添加列索引
    poly_df = pd.DataFrame(concatenated_array, columns=column_names_ploy)
    data_df = pd.concat([data_df, poly_df], axis=1)

    return data_df


def multi_variate_cross_generate_poly_features(df, fillCols, degree):
    data_df = df.copy()
    double_data_df = data_df[fillCols]
    poly_features = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    ploy_variable = poly_features.fit_transform(double_data_df)[:, len(fillCols):]
    column_temps = poly_features.get_feature_names_out(input_features=double_data_df.columns)[len(fillCols):]

    # 生成双变量多阶多项式衍升特征转换为DataFrame对象，并添加列索引
    poly_df = pd.DataFrame(ploy_variable, columns=column_temps)
    data_df = pd.concat([data_df, poly_df], axis=1)

    return data_df


def group_combination_statistic(df, statistics_colNames, groupby_colNames, statistical_methods):
    data_df = df.copy()

    df_temp_list = []
    for statistics_colName in statistics_colNames:
        if 'q1' in statistical_methods:
            data_df[f"{statistics_colName}_{'&'.join(groupby_colNames)}_q1"] = data_df[statistics_colNames].quantile(0.25)
            statistical_methods.remove('q1')
        if 'q2' in statistical_methods:
            data_df[f"{statistics_colName}_{'&'.join(groupby_colNames)}_q2"] = data_df[statistics_colNames].quantile(0.75)
            statistical_methods.remove('q2')

        group_features_df = data_df.groupby(groupby_colNames)[statistics_colName].agg(statistical_methods).reset_index()

        statistical_features_cols = []
        for statistical_method in statistical_methods:
            statistical_features_cols.append(f'{statistics_colName}_{"&".join(groupby_colNames)}_{statistical_method}')

        statistical_features_cols = groupby_colNames + statistical_features_cols
        group_features_df.columns = statistical_features_cols

        df_temp = pd.merge(data_df, group_features_df, how='left', on=groupby_colNames)
        df_temp_list.append(df_temp)

    if len(df_temp_list) == 1:
        df_combined = df_temp_list[0]
    else:
        df_combined = pd.merge(*df_temp_list, how='outer', on=df.columns.tolist())

    return df_combined


ori_df = pd.read_csv(r"E:\gitlocal\ml_code\common_dataset\Narrativedata.csv", index_col=0)

filled_df = miss_value_filler(ori_df,  # 最原始的DataF数据
                              delCols=None,  # 不需要数据编码的列，列表类型
                              fillCols=['Age'],  # 需要填补缺失值的列，列表类型
                              outlier=None,  # 异常值字典，字典类型{col:outlier}
                              fill_method="bfill",  # 填充方法，列表类型或字符串类型
                              model=RandomForestRegressor(n_estimators=100),  # 填补缺失值拟合模型
                              fill_mode="basic",  # 填充方法名称（选择哪种填充方式），字符串类型
                              n_clusters=2,  # K-means聚类算法的聚类中心数
                              draw=False)  # 是否绘制对比图

# data_1 = single_variate_generate_features(df=filled_df, fillCols=['Age'],
#                                           methods=["binarizer", "kbins", "single_variable_poly"],
#                                           threshold=30, n_bins=5, degree=5)
#
# data_2 = double_variate_cross_generate_math_features(df=filled_df,
#                                                      fillCols=['Sex', 'Embarked', 'Survived'],
#                                                      methods=["add", "sub", "div", "multi"])
#
# data_3 = double_variate_cross_generate_features(colNames=['Sex', 'Embarked', 'Survived'],
#                                                 df=filled_df, OneHot=True)
#
# data_4 = double_variate_cross_generate_poly_features(df=filled_df,
#                                                      fillCols=['Sex', 'Embarked', 'Survived'], degree=2)
#
# data_5 = multi_variate_cross_generate_features(colNames=['Sex', 'Embarked', 'Survived'],
#                                                df=filled_df, OneHot=True)
#
# data_6 = multi_variate_cross_generate_poly_features(df=filled_df,
#                                                     fillCols=['Sex', 'Embarked', 'Survived'], degree=3)

data_7 = group_combination_statistic(df=filled_df, statistics_colNames=['Age', "Survived"], groupby_colNames=['Sex', 'Embarked'],
                                     statistical_methods=['mean', 'median', 'var', 'max', 'min', 'std'])

# 设置显示所有列
pd.set_option('display.max_columns', None)
# print(data_1)
# print(data_2)
# print(data_2.isnull().sum())
# print(data_3)
# print(data_4)
# print(data_5)
# print(data_5.isnull().sum())
# print(data_6)
print(data_7)
print(data_7.isnull().sum())
