import pandas as pd

# 创建一个DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})

# 创建一个Series
series = pd.Series([9, 10, 11, 12])

# 将Series的值赋值给DataFrame的某列
df['B'] = series

# 打印更新后的DataFrame
print(df)
