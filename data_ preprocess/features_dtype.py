import pandas as pd

# 分类有序特征
OrderedClassify_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
                        'subsector_id', 'category_1',
                        'most_recent_sales_range', 'most_recent_purchases_range',
                        'category_4', 'city_id', 'state_id', 'category_2']
# 分类离散特征
DiscreteClassify_cols = []

# 连续数字特征
ContinuousNumeric_cols = {'numerical_1', 'numerical_2',
                          'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                          'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                          'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12'}

# 时间序列特征
Timeseries_cols = []

ori_df = pd.read_csv(r'E:\gitlocal\ml_code\ori_dataset\merchants.csv', header=0)
# 检验特征是否划分完全
assert len(OrderedClassify_cols) + len(DiscreteClassify_cols) + len(ContinuousNumeric_cols) + len(Timeseries_cols) == \
       ori_df.shape[1]
