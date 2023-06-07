# 示例4：包含日期和时间
from data_process.missvalue_filler import *

ori_df = pd.read_csv(r'E:\gitlocal\ml_code\ori_dataset\merchants.csv', header=0)
fillCols = ['merchant_group_id', 'merchant_category_id',
            'subsector_id', 'category_1',
            'most_recent_sales_range', 'most_recent_purchases_range',
            'category_4', 'city_id', 'state_id', 'category_2']

filled_df = miss_value_filler(ori_df,  # 最原始的DataF数据
                              delCols=None,  # 不需要数据编码的列，列表类型
                              fillCols=fillCols,  # 需要填补缺失值的列，列表类型
                              fill_method="ffill",  # 填充方法，列表类型或字符串类型
                              model=None,   # 填补缺失值拟合模型
                              fill_mode="sequential",  # 填充方法名称（选择哪种填充方式），字符串类型
                              n_clusters=5,  # K-means聚类算法的聚类中心数
                              draw=False)  # 是否绘制对比图
