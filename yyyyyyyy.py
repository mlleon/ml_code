import pandas as pd

# 创建一个示例Series
s = pd.Series([1, None, 3, None, 5, None])

# 获取只包含缺失值的行
missing_rows = s[s.isnull()]

print(missing_rows)
