"""
数据索引与选取
我们对DataFrame进行选择，大抵从这三个层次考虑：行列、区域、单元格。
其对应使用的方法如下：
一.行，列 --> df[]
二.区域 --> df.loc[], df.iloc[], df.ix[]
三.单元格 --> df.at[], df.iat[]

下面开始练习：
"""
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(6, 4), index=list('abcdef'), columns=list('ABCD'))
"""
1.df[]:
一维
行维度：整数切片、标签切片、 < 布尔数组 >
列维度：标签索引、标签列表、Callable
"""

df[:3]
df['a':'c']
df[[True, True, True, False, False, False]]  # 前三行（布尔数组长度等于行数）
df[df['A'] > 0]  # A列值大于0的行
df[(df['A'] > 0) | (df['B'] > 0)]  # A列值大于0，或者B列大于0的行
df[(df['A'] > 0) & (df['C'] > 0)]  # A列值大于0，并且C列大于0的行

df['A']
df[['A', 'B']]
df[lambda df: df.columns[0]]  # Callable

"""
2.df.loc[]
二维，先行后列
行维度：标签索引、标签切片、标签列表、 < 布尔数组 >、Callable
列维度：标签索引、标签切片、标签列表、 < 布尔数组 >、Callable
"""

df.loc['a', :]
df.loc['a':'d', :]
df.loc[['a', 'b', 'c'], :]
df.loc[[True, True, True, False, False, False], :]  # 前三行（布尔数组长度等于行数）
df.loc[df['A'] > 0, :]
df.loc[df.loc[:, 'A'] > 0, :]
df.loc[df.iloc[:, 0] > 0, :]
df.loc[lambda _df: _df.A > 0, :]

df.loc[:, 'A']
df.loc[:, 'A':'C']
df.loc[:, ['A', 'B', 'C']]
df.loc[:, [True, True, True, False]]  # 前三列（布尔数组长度等于行数）
df.loc[:, df.loc['a'] > 0]  # a行大于0的列
df.loc[:, df.iloc[0] > 0]  # 0行大于0的列
df.loc[:, lambda _df: ['A', 'B']]


df.A.loc[lambda s: s > 0]
"""
3.df.iloc[]
二维，先行后列
行维度：整数索引、整数切片、整数列表、 < 布尔数组 >
列维度：整数索引、整数切片、整数列表、 < 布尔数组 >、Callable

"""

df.iloc[3, :]
df.iloc[:3, :]
df.iloc[[0, 2, 4], :]
df.iloc[[True, True, True, False, False, False], :]  # 前三行（布尔数组长度等于行数）
df.iloc[df['A'] > 0, :]  # × 为什么不行呢？想不通！
df.iloc[df.loc[:, 'A'] > 0, :]  # ×
df.iloc[df.iloc[:, 0] > 0, :]  # ×
df.iloc[lambda _df: [0, 1], :]

df.iloc[:, 1]
df.iloc[:, 0:3]
df.iloc[:, [0, 1, 2]]
df.iloc[:, [True, True, True, False]]  # 前三列（布尔数组长度等于行数）
df.iloc[:, df.loc['a'] > 0]  # ×
df.iloc[:, df.iloc[0] > 0]  # ×
df.iloc[:, lambda _df: [0, 1]]

"""
4.df.ix[]
二维，先行后列
行维度：整数索引、整数切片、整数列表、标签索引、标签切片、标签列表、< 布尔数组 >、Callable
列维度：整数索引、整数切片、整数列表、标签索引、标签切片、标签列表、< 布尔数组 >、Callable
"""

df.ix[0, :]
df.ix[0:3, :]
df.ix[[0, 1, 2], :]

df.ix['a', :]
df.ix['a':'d', :]
df.ix[['a', 'b', 'c'], :]

df.ix[:, 0]
df.ix[:, 0:3]
df.ix[:, [0, 1, 2]]

df.ix[:, 'A']
df.ix[:, 'A':'C']
df.ix[:, ['A', 'B', 'C']]

"""
5.df.at[]
精确定位单元格
行维度：标签索引
列维度：标签索引
"""

df.at['a', 'A']

"""
6.df.iat[]
精确定位单元格
行维度：整数索引
列维度：整数索引
"""
df.iat[0, 0]




from pandas import DataFrame
import pandas as pd
import numpy as np

df = DataFrame(np.random.randn(4, 5), columns=['A', 'B', 'C', 'D', 'E'])
"""
DataFrame数据预览：
          A         B         C         D         E
0  0.673092  0.230338 -0.171681  0.312303 -0.184813
1 -0.504482 -0.344286 -0.050845 -0.811277 -0.298181
2  0.542788  0.207708  0.651379 -0.656214  0.507595
3 -0.249410  0.131549 -2.198480 -0.437407  1.628228
计算各列数据总和并作为新列添加到末尾
"""
df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
"""计算各行数据总和并作为新行添加到末尾"""
df.loc['Row_sum'] = df.apply(lambda x: x.sum())
"""
最终数据结果：
                A         B         C         D         E   Col_sum
0        0.673092  0.230338 -0.171681  0.312303 -0.184813  0.859238
1       -0.504482 -0.344286 -0.050845 -0.811277 -0.298181 -2.009071
2        0.542788  0.207708  0.651379 -0.656214  0.507595  1.253256
3       -0.249410  0.131549 -2.198480 -0.437407  1.628228 -1.125520
Row_sum  0.461987  0.225310 -1.769627 -1.592595  1.652828 -1.022097
"""