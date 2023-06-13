from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from missvalue_filler import *
import numpy as np
import itertools


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
        newNames = '&'.join([col + '_' + str(val) for col, val in zip(colNames, combination)])
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

    # 将列名中的空格替换为下划线
    data_df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    # 找到重复列的布尔索引
    duplicated_columns = data_df.T.duplicated()
    # 删除重复列
    data_df = data_df.loc[:, ~duplicated_columns]

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

    # 将列名中的空格替换为下划线
    data_df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    # 找到重复列的布尔索引
    duplicated_columns = data_df.T.duplicated()
    # 删除重复列
    data_df = data_df.loc[:, ~duplicated_columns]

    return data_df


def group_statistical_firstorder_features(df, statistics_colNames, groupby_colNames, statistical_methods):
    data_df = df.copy()

    df_temp_list = []
    stat_methods = statistical_methods.copy()
    for statistics_colName in statistics_colNames:
        if 'q1' in stat_methods:
            data_df[f"{statistics_colName}_{'&'.join(groupby_colNames)}_q1"] = data_df[statistics_colName].quantile(
                0.25)
            stat_methods.remove('q1')
        if 'q2' in stat_methods:
            data_df[f"{statistics_colName}_{'&'.join(groupby_colNames)}_q2"] = data_df[statistics_colName].quantile(
                0.75)
            stat_methods.remove('q2')

        group_features_df = data_df.groupby(groupby_colNames)[statistics_colName].agg(stat_methods).reset_index()

        statistical_features_cols = []
        for stat_method in stat_methods:
            statistical_features_cols.append(f'{statistics_colName}_{"&".join(groupby_colNames)}_{stat_method}')

        statistical_features_cols = groupby_colNames + statistical_features_cols
        group_features_df.columns = statistical_features_cols

        df_temp = pd.merge(data_df, group_features_df, how='left', on=groupby_colNames)
        df_temp_list.append(df_temp)
        stat_methods = statistical_methods.copy()

    if len(df_temp_list) == 1:
        df_combined = df_temp_list[0]
    else:
        df_combined = pd.merge(*df_temp_list, how='outer')

    return df_combined


# 分组统计衍生特征函数（二阶）
def group_statistical_secorder_features(features_df,
                                        groupby_columns,
                                        colNames_sub,
                                        statistical_methods,
                                        stream_smooth=True,
                                        good_combination=True,
                                        intra_group_normalization=True,
                                        gap_feature=True,
                                        data_skew_subtractor=True,
                                        data_skew_divider=True,
                                        variable_coefficient=True):
    """二阶衍生特征函数

    :param features_df: 传入一阶分组统计衍生函数[group_statistical_oneorder_features]分组列索引名称列表
    :param groupby_columns：传入一阶分组统计衍生函数[group_statistical_oneorder_features]分组列索引名称列表
    :param colNames_sub:传入一阶分组统计衍生函数[group_statistical_oneorder_features]被汇总的列索引名称列表
    :param statistical_methods:传入一阶分组统计衍生函数[group_statistical_oneorder_features]统计方法列表

    :param stream_smooth：是否执行流量平滑特征(stream_smooth)
    :param good_combination:是否执行黄金组合特征(good_combination)
    :param intra_group_normalization:是否执行组内归一化特征(Intra_group_normalization)
    :param gap_feature:是否执行Gap特征(gap_feature)
    :param data_skew_subtractor:是否执行数据倾斜(data_skew)
    :param data_skew_divider: 是否执行数据倾斜(data_skew)
    :param variable_coefficient:是否执行变异系数(variable_coefficient)

    :return：分组统计衍生特征（一阶衍生特征），分组统计衍生特征（一阶衍生特征） + 二阶衍升特征

    """

    # 分组统计衍生特征函数（一阶）
    def group_statistical_oneorder_features(data_df, groupby_colNames, stat_colNames, stat_Methods):
        """分组统计衍生函数

        :param data_df: 原始数据集
        :param groupby_colNames：传入分组列索引名称列表
        :param stat_colNames:传入统计计算的列索引名称列表
        :param stat_Methods:传入统计量方法列表

        :return：分组衍生后的新特征（一阶特征），新特征（一阶特征）列索引名称

        """

        aggs = {}
        groupby_features_df = None
        groupby_features_cols = groupby_colNames.copy()
        for col in stat_colNames:
            features_df_1 = data_df.copy()
            stat_list = stat_Methods.copy()
            if 'q1' in stat_Methods:
                features_df_1[f"{col}_{'&'.join(groupby_features_cols)}_q1"] = features_df_1[col].quantile(0.25)
                stat_list.remove('q1')
            if 'q2' in stat_Methods:
                features_df_1[f"{col}_{'&'.join(groupby_features_cols)}_q2"] = features_df_1[col].quantile(0.75)
                stat_list.remove('q2')

            # 获取分组特征新的列名称
            aggs[col] = stat_list
            groupby_columns_join = '&'.join(groupby_colNames)
            for key in aggs.keys():
                groupby_features_cols.extend([key + '_' + groupby_columns_join + '_' + stat for stat in stat_list])

            # 3.分组统计后的新特征对象，并添加新的特征列名称
            for key, value in aggs.items():
                if 'q1' in value:
                    value.remove('q1')
                if 'q2' in value:
                    value.remove('q2')
            features_df_2 = data_df.groupby(groupby_colNames)[col].agg(aggs[col]).reset_index()
            features_df_2.columns = groupby_features_cols
            groupby_features_df = pd.merge(features_df_1, features_df_2, on=groupby_colNames, how='outer')

        return groupby_features_df, groupby_features_df.columns

    def acquire_list_element(column_indexs_list, specific_statistical_method):
        """查找对应分组统计方法的特征名称

        :param column_indexs_list：传入分组统计衍生函数的返回值，groupby_features_cols
        :param specific_statistical_method:传入特定统计方法（如：'mean', 'min', 'max'等）

        :return：被查找的分组统计方法的特征名称（如：'MonthlyCharges_tenure&SeniorCitizen_mean'）

        """
        group_combination_statistic_names = []
        for col_name in column_indexs_list:
            if '_' not in col_name:
                continue
            if col_name[col_name.rindex('_') + 1:] == specific_statistical_method:
                group_combination_statistic_names.append(col_name)
        return group_combination_statistic_names

    # 执行一阶分组统计衍生函数
    group_features_df, col_Names_index = group_statistical_oneorder_features(features_df.copy(),
                                                                             groupby_columns,
                                                                             colNames_sub,
                                                                             statistical_methods)

    """1.原始特征与分组汇总特征交叉衍生"""
    # 1.1 流量平滑特征(stream_smooth)
    if stream_smooth:
        # 调用查找列表某个元素的值和对应的索引值
        stream_smooth_feature_names = acquire_list_element(col_Names_index, 'mean')

        # 计算流量平滑特征(stream_smooth)
        stream_smooth_dict = {}
        for stream_smooth_feature_name in stream_smooth_feature_names:
            stream_smooth_dict[stream_smooth_feature_name] = \
                group_features_df[groupby_columns].sum(axis=1) / (
                        group_features_df[stream_smooth_feature_name] + 1e-5)
        stream_smooth_feature = pd.DataFrame(stream_smooth_dict)

        # 流量平滑特征(stream_smooth)加上前缀stream_smooth_
        stream_smooth_feature = stream_smooth_feature.add_prefix('stream_smooth_')
        stream_smooth_feature = pd.concat([group_features_df, stream_smooth_feature], axis=1)
    else:
        stream_smooth_feature = group_features_df

    # 1.2 黄金组合特征(good_combination)
    if good_combination:
        # 调用查找列表某个元素的值和对应的索引值
        good_combination_feature_names = acquire_list_element(col_Names_index, 'mean')

        # 计算黄金组合特征(good_combination)
        good_combination_dict = {}
        for good_combination_feature_name in good_combination_feature_names:
            good_combination_dict[good_combination_feature_name] = \
                group_features_df[groupby_columns].sum(axis=1) - group_features_df[good_combination_feature_name]
        good_combination_feature = pd.DataFrame(good_combination_dict)

        # 黄金组合特征(good_combination)加上前缀good_combination_
        good_combination_feature = good_combination_feature.add_prefix('good_combination_')
        good_combination_feature = pd.concat([group_features_df, good_combination_feature], axis=1)
    else:
        good_combination_feature = group_features_df
    good_combination_feature = pd.merge(stream_smooth_feature, good_combination_feature, how='left')

    # 1.3 组内归一化特征(Intra_group_normalization)
    if intra_group_normalization:
        # 调用查找列表某个元素的值和对应的索引值
        intra_group_normalization_feature_means = acquire_list_element(col_Names_index, 'mean')
        # 组内归一化mean特征列(intra_group_normalization)
        intra_group_normalization_mean_dict = {}
        for intra_group_normalization_feature_mean in intra_group_normalization_feature_means:
            intra_group_normalization_mean_dict[intra_group_normalization_feature_mean] = \
                (group_features_df[groupby_columns].sum(axis=1) - group_features_df[
                    intra_group_normalization_feature_mean])
        intra_group_normalization_mean_feature = pd.DataFrame(intra_group_normalization_mean_dict)

        # 调用查找列表某个元素的值和对应的索引值
        intra_group_normalization_feature_vars = acquire_list_element(col_Names_index, 'var')
        # 组内归一化var特征列(intra_group_normalization)
        intra_group_normalization_var_dict = {}
        for intra_group_normalization_feature_var in intra_group_normalization_feature_vars:
            intra_group_normalization_var_dict[intra_group_normalization_feature_var] = \
                np.sqrt(group_features_df[intra_group_normalization_feature_var]) + 1e-5
        intra_group_normalization_var_feature = pd.DataFrame(intra_group_normalization_var_dict)
        intra_group_normalization_var_feature.columns = intra_group_normalization_mean_feature.columns

        # 计算组内归一化特征最终值
        intra_group_normalization_mean_var_feature = intra_group_normalization_mean_feature / intra_group_normalization_var_feature

        # 组内归一化特征(intra_group_normalization)加上前缀intra_group_normalization_和后缀_var
        intra_group_normalization_mean_var_feature = \
            intra_group_normalization_mean_var_feature.add_prefix('intra_group_normalization_')
        intra_group_normalization_mean_var_feature = \
            intra_group_normalization_mean_var_feature.add_suffix('_var')
        intra_group_normalization_mean_var_feature = \
            pd.concat([group_features_df, intra_group_normalization_mean_var_feature], axis=1)
    else:
        intra_group_normalization_mean_var_feature = group_features_df
    intra_group_normalization_mean_var_feature = \
        pd.merge(good_combination_feature, intra_group_normalization_mean_var_feature, how='left')

    """2.分组汇总特征彼此交叉衍生"""
    # 2.1 Gap特征(gap_feature)
    if gap_feature:
        # 调用查找列表某个元素的值和对应的索引值
        gap_feature_q2s = acquire_list_element(col_Names_index, specific_statistical_method='q2')
        # Gap的q2特征列(gap)
        gap_q2_dict = {}
        for gap_feature_q2 in gap_feature_q2s:
            gap_q2_dict[gap_feature_q2] = group_features_df[gap_feature_q2]
        gap_q2_feature = pd.DataFrame(gap_q2_dict)

        # 调用查找列表某个元素的值和对应的索引值
        gap_feature_q1s = acquire_list_element(col_Names_index, specific_statistical_method='q1')
        # Gap的q1特征列(gap)
        gap_q1_dict = {}
        for gap_feature_q1 in gap_feature_q1s:
            gap_q1_dict[gap_feature_q1] = group_features_df[gap_feature_q1]
        gap_q1_feature = pd.DataFrame(gap_q1_dict)
        gap_q1_feature.columns = gap_q2_feature.columns

        # 计算Gap特征最终值
        gap_q2_q1_feature = gap_q2_feature - gap_q1_feature

        # Gap特征(gap)加上前缀gap_和后缀_q1
        gap_q2_q1_feature = gap_q2_q1_feature.add_prefix('gap_')
        gap_q2_q1_feature = gap_q2_q1_feature.add_suffix('_q1')
        gap_q2_q1_feature = pd.concat([group_features_df, gap_q2_q1_feature], axis=1)
    else:
        gap_q2_q1_feature = group_features_df
    gap_q2_q1_feature = pd.merge(intra_group_normalization_mean_var_feature, gap_q2_q1_feature, how='left')

    # 2.2 数据倾斜特征(data_skew_subtractor)
    if data_skew_subtractor:
        # 调用查找列表某个元素的值和对应的索引值
        data_skew_feature_means = acquire_list_element(col_Names_index, specific_statistical_method='mean')
        # 数据倾斜mean特征列(data_skew)
        data_skew_mean_dict = {}
        for data_skew_feature_mean in data_skew_feature_means:
            data_skew_mean_dict[data_skew_feature_mean] = group_features_df[data_skew_feature_mean]
        data_skew_mean_feature = pd.DataFrame(data_skew_mean_dict)

        # 调用查找列表某个元素的值和对应的索引值
        data_skew_feature_medians = acquire_list_element(col_Names_index, specific_statistical_method='median')
        # 数据倾斜median特征列(data_skew)
        data_skew_median_dict = {}
        for data_skew_feature_median in data_skew_feature_medians:
            data_skew_median_dict[data_skew_feature_median] = group_features_df[data_skew_feature_median]
        data_skew_median_feature = pd.DataFrame(data_skew_median_dict)
        data_skew_median_feature.columns = data_skew_mean_feature.columns

        # 计算数据倾斜特征最终值(减法)
        data_skew_mean_median_feature = data_skew_mean_feature - data_skew_median_feature

        # 数据倾斜特征(data_skew)加上前缀data_skew_subtractor_和后缀_median
        data_skew_subtractor_mean_median_feature = data_skew_mean_median_feature.add_prefix('data_skew_subtractor_')
        data_skew_subtractor_mean_median_feature = data_skew_subtractor_mean_median_feature.add_suffix('_median')
        data_skew_subtractor_mean_median_feature = pd.concat(
            [group_features_df, data_skew_subtractor_mean_median_feature], axis=1)
    else:
        data_skew_subtractor_mean_median_feature = group_features_df
    data_skew_subtractor_mean_median_feature = pd.merge(gap_q2_q1_feature, data_skew_subtractor_mean_median_feature,
                                                        how='left')

    # 2.3 数据倾斜特征(data_skew_divider)
    if data_skew_divider:
        # 调用查找列表某个元素的值和对应的索引值
        data_skew_feature_means = acquire_list_element(col_Names_index, 'mean')
        # 数据倾斜mean特征列(data_skew)
        data_skew_mean_dict = {}
        for data_skew_feature_mean in data_skew_feature_means:
            data_skew_mean_dict[data_skew_feature_mean] = group_features_df[data_skew_feature_mean]
        data_skew_mean_feature = pd.DataFrame(data_skew_mean_dict)

        # 调用查找列表某个元素的值和对应的索引值
        data_skew_feature_medians = acquire_list_element(col_Names_index, 'median')
        # 数据倾斜median特征列(data_skew)
        data_skew_median_dict = {}
        for data_skew_feature_median in data_skew_feature_medians:
            data_skew_median_dict[data_skew_feature_median] = group_features_df[data_skew_feature_median] + 1e-5
        data_skew_median_feature = pd.DataFrame(data_skew_median_dict)
        data_skew_median_feature.columns = data_skew_mean_feature.columns

        # 计算数据倾斜特征最终值(除法)
        data_skew_mean_median_feature = data_skew_mean_feature / data_skew_median_feature

        # 数据倾斜特征(data_skew)加上前缀data_skew_divider_和后缀_median
        data_skew_divider_mean_median_feature = data_skew_mean_median_feature.add_prefix('data_skew_divider_')
        data_skew_divider_mean_median_feature = data_skew_divider_mean_median_feature.add_suffix('_median')
        data_skew_divider_mean_median_feature = pd.concat(
            [group_features_df, data_skew_divider_mean_median_feature], axis=1)
    else:
        data_skew_divider_mean_median_feature = group_features_df
    data_skew_divider_mean_median_feature = pd.merge(data_skew_subtractor_mean_median_feature,
                                                     data_skew_divider_mean_median_feature, how='left')

    # 2.4 变异系数特征(variable_coefficient)
    if variable_coefficient:
        # 调用查找列表某个元素的值和对应的索引值
        variable_coefficient_feature_vars = acquire_list_element(col_Names_index, 'var')
        # 变异系数var特征列(variable_coefficient)
        variable_coefficient_var_dict = {}
        for variable_coefficient_feature_var in variable_coefficient_feature_vars:
            variable_coefficient_var_dict[variable_coefficient_feature_var] = np.sqrt(
                group_features_df[variable_coefficient_feature_var])
        variable_coefficient_var_feature = pd.DataFrame(variable_coefficient_var_dict)

        # 调用查找列表某个元素的值和对应的索引值
        variable_coefficient_feature_means = acquire_list_element(col_Names_index, 'mean')
        # 变异系数mean特征列(variable_coefficient)
        variable_coefficient_mean_dict = {}
        for variable_coefficient_feature_mean in variable_coefficient_feature_means:
            variable_coefficient_mean_dict[variable_coefficient_feature_mean] = group_features_df[
                                                                                    variable_coefficient_feature_mean] + 1e-10
        variable_coefficient_mean_feature = pd.DataFrame(variable_coefficient_mean_dict)
        variable_coefficient_mean_feature.columns = variable_coefficient_var_feature.columns

        # 计算变异系数特征最终值
        variable_coefficient_var_mean_feature = variable_coefficient_var_feature / variable_coefficient_mean_feature

        # 变异系数特征(variable_coefficient)加上前缀variable_coefficient_和后缀_mean
        variable_coefficient_var_mean_feature = variable_coefficient_var_mean_feature.add_prefix(
            'variable_coefficient_')
        variable_coefficient_var_mean_feature = variable_coefficient_var_mean_feature.add_suffix('_mean')
        variable_coefficient_var_mean_feature = pd.concat(
            [group_features_df, variable_coefficient_var_mean_feature], axis=1)
    else:
        variable_coefficient_var_mean_feature = group_features_df
    groupby_features_all_df = pd.merge(data_skew_divider_mean_median_feature, variable_coefficient_var_mean_feature,
                                       how='left')

    return groupby_features_all_df


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


def gen_features(methods):
    data = {}

    feature_methods = {
        "single_variate_generate_features": lambda: single_variate_generate_features(
            df=filled_df,
            fillCols=['Age'],
            methods=["binarizer", "kbins", "single_variable_poly"],
            threshold=30,
            n_bins=5,
            degree=5),
        "double_variate_cross_generate_math_features": lambda: double_variate_cross_generate_math_features(
            df=filled_df,
            fillCols=['Sex', 'Embarked', 'Survived'],
            methods=["add", "sub", "div", "multi"]),
        "double_variate_cross_generate_features": lambda: double_variate_cross_generate_features(
            colNames=['Sex', 'Embarked', 'Survived'],
            df=filled_df,
            OneHot=True),
        "double_variate_cross_generate_poly_features": lambda: double_variate_cross_generate_poly_features(
            df=filled_df,
            fillCols=['Sex', 'Embarked', 'Survived'],
            degree=2),
        "multi_variate_cross_generate_features": lambda: multi_variate_cross_generate_features(
            colNames=['Sex', 'Embarked', 'Survived'],
            df=filled_df,
            OneHot=True),
        "multi_variate_cross_generate_poly_features": lambda: multi_variate_cross_generate_poly_features(
            df=filled_df,
            fillCols=['Sex', 'Embarked', 'Survived'],
            degree=3),
        "group_statistical_firstorder_features": lambda: group_statistical_firstorder_features(
            df=filled_df,
            statistics_colNames=['Age', "Survived"],
            groupby_colNames=['Sex', 'Embarked'],
            statistical_methods=['mean', 'median', 'var', 'max', 'min', 'std', 'q1', 'q2']),
        "group_statistical_secorder_features": lambda: group_statistical_secorder_features(
            features_df=filled_df,
            groupby_columns=['Sex', 'Embarked'],
            colNames_sub=['Survived'],
            statistical_methods=['mean', 'median', 'var', 'max', 'min', 'std', 'q1', 'q2'],
            stream_smooth=True,
            good_combination=False,
            intra_group_normalization=False,
            gap_feature=False,
            data_skew_subtractor=False,
            data_skew_divider=False,
            variable_coefficient=False)
    }

    for method in methods:
        if method in feature_methods:
            data[method] = feature_methods[method]()
        else:
            print("Invalid feature generate method:", method)

    features_list = [value for value in data.values()]

    # 初始化合并结果为第一个DataFrame对象
    combined_df = features_list[0]
    # 从第二个DataFrame开始迭代合并
    for df in features_list[1:]:
        # 检查列名是否有重复
        if combined_df.columns.duplicated().any() or df.columns.duplicated().any():
            raise ValueError("Duplicate column names found!")
        combined_df = pd.merge(combined_df, df, how='outer')

    return combined_df


gen_features(methods=['single_variate_generate_features'
    , 'double_variate_cross_generate_math_features'
    , 'double_variate_cross_generate_features'
    , 'double_variate_cross_generate_poly_features'
    , 'multi_variate_cross_generate_features'
    , 'multi_variate_cross_generate_poly_features'
    # ,'group_statistical_firstorder_features'
    ])
