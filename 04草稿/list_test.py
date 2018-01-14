plist = [x for x in range(1,100)]
pplist = [x for x in range(10)]
# print( plist[::3])
# print(pplist)


from itertools import groupby

lst= [
    2648, 2648, 2648, 63370, 63370, 425, 425, 120,
    120, 217, 217, 189, 189, 128, 128, 115, 115, 197,
    19752, 152, 152, 275, 275, 1716, 1716, 131, 131,
    98, 98, 138, 138, 277, 277, 849, 302, 152, 1571,
    68, 68, 102, 102, 92, 92, 146, 146, 155, 155,
    9181, 9181, 474, 449, 98, 98, 59, 59, 295, 101, 5,11
]

for i in range(max(lst)/10):
    a= i*10
    b= (i+1)*10
    print('{}-{}: {} '.format(a, b, len([pp for pp in lst if pp>=a and pp<b]) ))

# for k, g in groupby(sorted(lst), key=lambda x: x//50):
#     print(len(list(g)))
#     print('{}-{}: {}   =={}'.format(k*50, (k+1)*50-1, len(list(g)), 'AAAA' ))
#     print([ x for x in g])
#
# def group(lst, n):
#     num = len(lst) % n
#     print(num)
#     zipped = zip(*[iter(lst)] * n)
#     return zipped if not num else zipped + [lst[-num:], ]
# #
# # tmp = group(lst,3)
# # print(tmp[0])
# tmp = [iter(lst)] * 2
# print([x for x in tmp])


