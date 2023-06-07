import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_process.missvalue_filler import CategoricalMissingValueFiller
from data_process.missvalue_filler import ContinuousMissingValueFiller, ordered_encode_columns, one_hot_encode_columns

# 导入数据

merchant = pd.read_csv(r'E:\gitlocal\ml_code\ori_dataset\merchants.csv', header=0)
cols = ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']

df = ordered_encode_columns(merchant)