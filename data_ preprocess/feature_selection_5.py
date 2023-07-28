import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


def find_interval_with_max_value(x_values, y_values):
    """
    :param x_values（list）：这是一个包含折线图中所有 x 坐标值的列表。列表中的元素应按照 x 坐标的顺序排列，以确保方法的正确性。
    :param y_values（list）：这是一个包含与 x_values 列表中的每个 x 坐标对应的折线图数据值的列表。
    """
    max_value_index = y_values.index(max(y_values))  # 找到最大值的索引
    max_x_index = x_values[max_value_index]  # 获取最大值对应的 x 坐标
    max_y_value = max(y_values)

    # 找到最大值的左边界和右边界的索引
    left_index = max_value_index - 1 if max_value_index > 0 else 0
    right_index = max_value_index + 1 if max_value_index < len(x_values) - 1 else len(x_values) - 1

    # 获取最大值所在的区间
    interval_start = x_values[left_index]
    interval_end = x_values[right_index]

    return interval_start, interval_end, max_x_index, max_y_value


def variance_filter(x, threshold=0.8):
    x_bvar = VarianceThreshold(threshold * (1 - threshold)).fit_transform(x)
    x_fsvar = VarianceThreshold(np.median(x_bvar.var())).fit_transform(x_bvar)
    return x_fsvar


def chi2_filter(x, y):
    chivalues, pvalues_chi = chi2(x, y)
    k = chivalues.shape[0] - (pvalues_chi > 0.05).sum()
    return SelectKBest(chi2, k=k).fit_transform(x, y)


def f_classif_filter(x, y):
    F, pvalues = f_classif(x, y)
    k = F.shape[0] - (pvalues > 0.05).sum()
    return SelectKBest(f_classif, k=k).fit_transform(x, y)


def mutual_info_classif_filter(x, y):
    result = MIC(x, y)
    k = result.shape[0] - sum(result < 0)
    return SelectKBest(MIC, k=k).fit_transform(x, y)


def embedded_filter(x, y):
    # 3.1 训练一个简单的随机森林模型，用于得出绘制学习曲线的最大特征权重值
    RFC_ = RFC(n_estimators=10, random_state=0)
    rfc = RFC_.fit(x, y)
    # 3.2 获取简单随机森林最重要特征权重值
    max_feature_importance = RFC_.feature_importances_.max()

    # 4.1绘制不同特征权重阀值的学习曲线（0~最重要特征权重值）找出超参数threshold的一个范围
    thersholds = np.linspace(0, max_feature_importance, 20)
    scores = []
    for i in thersholds:
        x_embedded = SelectFromModel(rfc, threshold=i).fit_transform(x, y)
        score = cross_val_score(rfc, x_embedded, y, cv=10).mean()
        scores.append(score)

    plt.figure(figsize=(16, 8))
    plt.plot(thersholds, scores)
    plt.show()

    l_index, r_index = find_interval_with_max_value(thersholds, scores)[:2]

    # 4.2 根据4.1得出的最优值，缩小超参数threshold的取值范围，重新绘制学习曲线
    scores_ = []
    for i in np.linspace(round(l_index, 5), round(r_index, 5), 20):
        x_embedded_ = SelectFromModel(rfc, threshold=i).fit_transform(x, y)
        score_ = cross_val_score(rfc, x_embedded_, y, cv=10).mean()
        scores_.append(score_)

    plt.figure(figsize=(16, 8))
    plt.plot(np.linspace(round(l_index, 5), round(r_index, 5), 20), scores_)
    plt.xticks(np.linspace(round(l_index, 5), round(r_index, 5), 20))
    plt.show()

    # 使用 y 中的最大值对应索引找到对应的横坐标
    x_at_max_y, max_y_value = find_interval_with_max_value(np.linspace(round(l_index, 5), round(r_index, 5), 20),
                                                           scores_)[2:]
    feature = SelectFromModel(RFC_, threshold=round(x_at_max_y * 10 ** 8) / 10 ** 8).fit_transform(x, y)
    return feature


def wrapper_filter(x, y):
    # 20个特征为一个阶段，每次略掉20个特征，绘制n_features_to_select参数学习曲线
    RFC_ = RFC(n_estimators=10, random_state=0)

    score_ = []
    for i in range(1, (divmod(x.shape[1], 20)[0] + 1) * 20, divmod(x.shape[1], 20)[0]):
        X_wrapper = RFE(RFC_, n_features_to_select=i, step=50).fit_transform(x, y)
        once_ = cross_val_score(RFC_, X_wrapper, y, cv=5).mean()
        score_.append(once_)

    plt.figure(figsize=[20, 5])
    plt.plot(range(1, (divmod(x.shape[1], 20)[0] + 1) * 20, divmod(x.shape[1], 20)[0]), score_)
    plt.xticks(range(1, (divmod(x.shape[1], 20)[0] + 1) * 20, divmod(x.shape[1], 20)[0]))
    plt.show()

    # 使用 y 中的最大值对应索引找到对应的横坐标,选择保留分数最高的特征个数
    x_index = range(1, (divmod(x.shape[1], 20)[0] + 1) * 20, divmod(x.shape[1], 20)[0])
    x_at_max_y = x_index[np.argmax(score_)]
    feature = RFE(RFC_, n_features_to_select=x_at_max_y, step=20).fit_transform(x, y)
    return feature


def feature_select(data, VarianceFilter=True, Chi2Filter=False, f_classifFilter=False,
                   mutual_info_classifFilter=False, EmbeddedFilter=False, WrapperFilter=False):
    # Split the dataset into features and labels
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    if VarianceFilter:
        x = VarianceThreshold().fit_transform(x)
        x = variance_filter(x)

    if Chi2Filter:
        x = VarianceThreshold().fit_transform(x)
        x = chi2_filter(x, y)

    if f_classifFilter:
        x = VarianceThreshold().fit_transform(x)
        x = f_classif_filter(x, y)

    if mutual_info_classifFilter:
        x = VarianceThreshold().fit_transform(x)
        x = mutual_info_classif_filter(x, y)

    if EmbeddedFilter:
        x = embedded_filter(x, y)

    if WrapperFilter:
        x = wrapper_filter(x, y)

    return x


data = pd.read_csv(r'E:\gitlocal\ml_code\common_dataset\DigitRecognizor.csv')

feature = feature_select(data, VarianceFilter=False, Chi2Filter=False, f_classifFilter=False,
                         mutual_info_classifFilter=True, EmbeddedFilter=False, WrapperFilter=False)

print(feature.shape[1])
