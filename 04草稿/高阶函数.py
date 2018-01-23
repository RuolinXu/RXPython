from functools import reduce

def foo(x, y):
    print('x={} y={}'.format(x, y))
    return x+y         # 返回值将作为下次的x值

lis = [x for x in range(20)]

aa = reduce(foo, lis)
aa = reduce(foo, lis, 100)
print(aa)


bb = map(lambda x: x % 2, range(7))
print(bb)