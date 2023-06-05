import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_process.missvalue_filler import CategoricalMissingValueFiller

# 导入数据
merchant = pd.read_csv('/home/leon/gitlocal/ml_code/ori_dataset/merchants.csv', header=0)
obj = CategoricalMissingValueFiller(merchant)
cols = ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
# 设置显示所有列
pd.set_option('display.max_columns', None)
df = obj.change_object_cols(cols)

print(df.head(5))