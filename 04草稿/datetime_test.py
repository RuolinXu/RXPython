import datetime

# now = datetime.datetime.now()
#
# # datetime => string
# print(now.strftime('%Y-%m-%d %H:%M:%S'))
#
# # string => datetime
# t_str = '2015-04-07 19:11:21'
# dt = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
#
# # timedelta 参数可正可负，int或float 支持加减乘除
# td = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
# td2 = td * 10
# print(td.days)
#
# dt2 = dt + td



start_time = datetime.datetime.now()

for x in range(1000000):
    pass

end_time = datetime.datetime.now()

print((end_time - start_time).microseconds)