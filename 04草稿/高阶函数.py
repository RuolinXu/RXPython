def test_map():
    def hoo(x):
        return x * x

    # bb = map(lambda x: x % 2, range(7))
    bb = map(hoo, range(7))
    print([x for x in bb])


def test_sort():
    aa = sorted([1, 66, 33, 22, 11], reverse=True)
    aa = sorted([1, 66, 33, 22, 11], key=lambda x: x % 2)
    print(aa)

    def foo(x):
        return x % 3

    aa1 = sorted([1, 66, 33, 22, 11], key=foo)
    print(aa1)

    list1 = [('david', 90), ('mary', 90), ('sara', 80), ('lily', 95)]
    bb = sorted(list1, key=lambda x: x[0], reverse=True)
    print(bb)


def test_reduce():
    from functools import reduce

    def foo(x, y):
        print('x={} y={}'.format(x, y))
        return x+y         # 返回值将作为下次的x值

    lis = [x for x in range(20)]

    # aa = reduce(foo, lis)
    aa = reduce(foo, lis, 100)
    print(aa)


def test_filter():
    import math

    def is_sqr(x):
        return math.sqrt(x) % 1 == 0

    aa = filter(is_sqr, range(1, 101))
    print([x for x in aa])

    def is_not_empty(s):
        return s and len(s.strip()) > 0

    bb = filter(is_not_empty, ['test', None, '', 'str', '  ', 'END'])
    print([x for x in bb])


if __name__ == '__main__':
    # test_map()
    test_sort()
    # test_reduce()
    # test_filter()


