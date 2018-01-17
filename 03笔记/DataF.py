import pandas as pd
import numpy as np

data = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
data2 = [{'a': 1, 'b': '2017-01-01', 'c': 3, 'd': 4},
         {'a': 5, 'b': '2017-01-02', 'c': 7, 'd': 8},
         {'a': 9, 'b': '2017-01-03', 'c': 11, 'd': 12}]
# df = pd.DataFrame(data2, columns=['a','b','c','d'])   # 用普通列表构造
df = pd.DataFrame(data2)     # 用字典列表构造

print('打印DataFrame')
print(df)

df1 = df.set_index(['b'])        # 设置索引列
print('打印设置了索引的DataFrame')
print(df1)

print('打印a列')
print(df1['a'])                  # 显示a列
print('打印某一索引行的a列')
print(df1.loc['2017-01-01']['a'])
print('打印某一索引行的a c 列')
print(df1.loc['2017-01-01', ['a', 'c']])
print('使用布尔索引')
print(df1[df1.a > 5])            # 布尔索引
print('对水平轴统计')
print(df1.mean(0))    # 对水平轴统计
print(df1['a'].mean(0))    # 对纵轴统计

print('按a列排序')
print(df1.sort_values('a', ascending=False))

print('按索引排序')
print(df1.sort_index(ascending=False))


# DataFrame.iterrows() 迭代(iterate)覆盖整个DataFrame的行中，返回(index, Series)对
arr1 = [x[1]['a'] for x in df1.iterrows()]
print(arr1)

for x in df1['a']:
    print(x)

df = pd.DataFrame({'A':['foo','bar','foo','bar','foo','bar','foo','bar'],
                   'B':['one','one','two','one','one','two','one','two'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)
                   })
print(df)
print('分组统计')
print(df.groupby(['A','B']).sum())




