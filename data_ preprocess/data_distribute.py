import csv
from data_process.datainfo_check import *


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime


# 将时间字符串转化为时间序列格式
def select_dates_(dates, num_dates):

    # 转换日期字符串为日期对象
    date_objects = [datetime.strptime(date, "%Y-%m") for date in dates]

    # 对日期对象进行排序
    sorted_dates = sorted(date_objects)

    # 计算平均间隔
    interval = (sorted_dates[-1] - sorted_dates[0]) / num_dates

    # 选择日期
    selected_dates = [sorted_dates[0] + i * interval for i in range(num_dates)]

    # 将日期对象转换回日期字符串
    selected_dates_str = [date.strftime("%Y-%m") for date in selected_dates]

    return selected_dates_str


# 判断元素是否是一个时间序列
def is_time_series(element):
    try:
        datetime.strptime(element, '%Y-%m')
        return True
    except ValueError:
        return False


def select_dates(dates, num_dates):
    date_objects = [datetime.strptime(date, "%Y-%m") for date in dates]
    sorted_dates = sorted(date_objects)
    interval = (sorted_dates[-1] - sorted_dates[0]) / num_dates
    selected_dates = [sorted_dates[0] + i * interval for i in range(num_dates)]
    selected_positions = [i for i in range(num_dates)]
    return selected_positions


def plot_column(df, col):
    column = df[col]
    plt.title(col)

    if column.dtype == 'int64':
        column.plot(kind='bar')
        if len(column.unique()) > 15:
            step = int(len(column.unique()) / 15)
            ticks = column.unique()[::step]
            plt.xticks(ticks)
    elif column.dtype == 'object':
        column.value_counts().plot(kind='bar')
        if not is_time_series(column.values[0]):
            # positions = select_dates(column.values, 15)
            # ticks = select_dates(column.values, 15)
            # plt.xticks(positions, ticks)
            if len(column.unique()) > 15:
                step = int(len(column.unique()) / 15)
                ticks = column.unique()[::step]
                plt.xticks(ticks)
    elif column.dtype == 'float64':
        column.plot(kind='density')
        if len(column.unique()) > 15:
            step = int(len(column.unique()) / 15)
            ticks = column.unique()[::step]
            plt.xticks(ticks)
    else:
        print(f"Unsupported data type: {column.dtype}")

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# 查看训练集不符合数据分布的的异常标签占比
def calculate_outliers(df, label_col="target"):
    target_values = df[label_col]
    mean = np.mean(target_values)
    std = np.std(target_values)
    threshold = abs(mean - 3 * std)
    outliers_count = np.sum(np.logical_or(target_values < -threshold, target_values > threshold))
    percentage = round(outliers_count / target_values.shape[0] * 100, 2)
    outliers_percentage = f"{percentage:.2f}%"

    # # 查看训练集不符合数据分布的的异常标签占比
    # count, percentage = calculate_outliers(data, label_col)
    # logger.info("训练集偏离数据分布的的异常标签个数和百分比：\n{}, {}".format(count, percentage))
    # sns.set()   # 如果是label数据是连续变量，借助概率密度直方图进行分布的观察
    # sns.histplot(data[label_col], kde=True)
    return outliers_count, outliers_percentage


def find_matching_rows(file1, file2, column_name):
    # 读取第一个CSV文件，提取column_name列的元素到一个集合中
    set1 = set()
    with open(file1, 'r') as f1:
        reader1 = csv.DictReader(f1)
        for row in reader1:
            set1.add(row[column_name])

    matching_rows = []
    # 读取第二个CSV文件，检查column_name列的元素是否存在于集合中
    with open(file2, 'r') as f2:
        reader2 = csv.DictReader(f2)
        for row in reader2:
            if row[column_name] in set1:
                matching_rows.append(row)

    return matching_rows


# 特征合并，用于多变量联合分布分析
def combine_feature(df):
    cols = df.columns
    feature1 = df[cols[0]].astype(str).values.tolist()
    feature2 = df[cols[1]].astype(str).values.tolist()
    return pd.Series([feature1[i] + '&' + feature2[i] for i in range(df.shape[0])])


# 检查数据集训练集和测试集分布是否一致
def check_data_distribution(train_path, test_path, check_features, label_col="target"):
    logfile_path = "dataset_check.log"
    logger = configure_logger(logfile_path)

    # 设置显示所有列
    pd.set_option('display.max_columns', None)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 查看训练集不符合数据分布的的异常标签占比
    count, percentage = calculate_outliers(train, label_col)
    logger.info("训练集不符合数据分布的的异常标签数量：\n {}  \n训练集不符合数据分布的的异常标签百分比：\n {}".format(count, percentage))

    # 检验训练集和测试集唯一id是否重叠
    if test['card_id'].nunique() + train['card_id'].nunique() != len(
            set(test['card_id'].values.tolist() + train['card_id'].values.tolist())):
        find_matching_rows(train_path, test_path, 'card_id')

    # 规律一致性分析
    """
            所谓规律一致性，指的是需要对训练集和测试集特征数据的分布进行简单比对，以“确定”两组数据是否诞生于同一个总体，
        即两组数据是否都遵循着背后总体的规律，即两组数据是否存在着规律一致性。
            我们知道，尽管机器学习并不强调样本-总体的概念，但在训练集上挖掘到的规律要在测试集上起到预测效果，
        就必须要求这两部分数据受到相同规律的影响。一般来说，对于标签未知的测试集，我们可以通过特征的分布规律来判断两组数据是否取自同一总体。

        单变量分析:
            首先我们先进行简单的单变量分布规律的对比。由于数据集中四个变量都是离散型变量，
            因此其分布规律我们可以通过相对占比分布（某种意义上来说也就是概率分布）来进行比较。

        多变量联合分布:
            所谓联合概率分布，指的是将离散变量两两组合，然后查看这个新变量的相对占比分布。例如特征1有0/1两个取值水平，
            特征2有A/B两个取值水平，则联合分布中就将存在0A、0B、1A、1B四种不同取值水平，然后进一步查看这四种不同取值水平出现的分布情况。

        规律一致性分析的实际作用
                在实际建模过程中，规律一致性分析是非常重要但又经常容易被忽视的一个环节。通过规律一致性分析，
            我们可以得出非常多的可用于后续指导后续建模的关键性意见。通常我们可以根据规律一致性分析得出以下基本结论：
                (1).如果分布非常一致，则说明所有特征均取自同一整体，训练集和测试集规律拥有较高一致性，模型效果上限较高，
            建模过程中应该更加依靠特征工程方法和模型建模技巧提高最终预测效果；
                (2).如果分布不太一致，则说明训练集和测试集规律不太一致，此时模型预测效果上限会受此影响而被限制，并且模型大概率容易过拟合，
            在实际建模过程中可以多考虑使用交叉验证等方式防止过拟合，并且需要注重除了通用特征工程和建模方法外的trick的使用；
    """
    # 单变量分析
    for feature in check_features:
        (train[feature].value_counts().sort_index() / train.shape[0]).plot()
        (test[feature].value_counts().sort_index() / test.shape[0]).plot()
        plt.legend(['train', 'test'])
        plt.xlabel(feature)
        plt.ylabel('ratio')
        plt.show()

    # 多变量联合分布
    n = len(check_features)
    for i in range(n - 1):
        for j in range(i + 1, n):
            cols = [check_features[i], check_features[j]]
            print(cols)
            train_dis = combine_feature(train[cols]).value_counts().sort_index() / train.shape[0]
            test_dis = combine_feature(test[cols]).value_counts().sort_index() / test.shape[0]
            index_dis = pd.Series(train_dis.index.tolist() + test_dis.index.tolist()).drop_duplicates().sort_values()
            (index_dis.map(train_dis).fillna(0)).plot()
            (index_dis.map(train_dis).fillna(0)).plot()
            plt.legend(['train', 'test'])
            plt.xlabel('&'.join(cols))
            plt.ylabel('ratio')
            plt.show()
