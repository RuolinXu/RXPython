a = {}

a[(-50,'2018-01-01')] = (110, '2018-01-01')

print(a.setdefault('default'))
print((-50,'2018-01-01') in a)
print('aaaa' in a)