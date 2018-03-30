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


#
# start_time = datetime.datetime.now()
#
# for x in range(1000000):
#     pass
#
# end_time = datetime.datetime.now()
#
# print((end_time - start_time).microseconds)


def datetime_int(datetime):
    import time
    print(datetime)
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(timeArray))

def int_datetime(timeStamp):
    import time
    timeArray = time.localtime(timeStamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", timeArray)


print(int_datetime(1493602260))
print(int_datetime(1246406400))
print(datetime_int('2017-05-01 09:31:00'))
