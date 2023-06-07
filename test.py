import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_process.missvalue_filler import CategoricalMissingValueFiller
from data_process.missvalue_filler import ContinuousMissingValueFiller, ordered_encode_columns, one_hot_encode_columns

# 导入数据
merchant = pd.read_csv(r'E:\gitlocal\ml_code\Narrativedata.csv', header=0)
cols = ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']


fill_methods = ['mean', 'mode', 'median',
                                                            'bfill', 'backfill', 'ffill',
                                                            'nearest', 'zero',
                                                            'slinear', 'linear',
                                                            'quadratic', 'cubic']
# merchant["Embarked"] = merchant["Embarked"].fillna("-1")
# df = one_hot_encode_columns(merchant, fillCols=["Embarked"])
df = ordered_encode_columns(merchant, fillCols=["Embarked"])
obj1 = CategoricalMissingValueFiller(df, fillCols=["Embarked"])
# obj1.kmeans_group_fill_method(corrCol="Sex_male")
# obj1.basic_fill_method(fill_methods=["bfill"])
obj1.sequential_kmeans_group_fill_method(clusterCol="Age", fillCols=["Embarked"])

# 设置显示所有列
pd.set_option('display.max_columns', None)