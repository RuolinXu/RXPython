import redis


class RedisHelper:
    def __init__(self, host='127.0.0.1', port=6379):
        self.__redis = redis.StrictRedis(host, port)

    def get(self, key):
        if self.__redis.exists(key):
            return self.__redis.get(key)
        else:
            return ""

    def set(self, key, value):
        self.__redis.set(key, value)

    def hasKey(self, key):
        return self.__redis.exists(key)

    def clear(self):
        return self.__redis.flushall()



if __name__ == '__main__':
    r = RedisHelper()
    a = [1,2,3,4,5,6,7]
    b="hellllllll"
    r.set('kkk', a)
    r.clear()
    f = r.get('kkk')
    print(r.hasKey('kkk'))
    print(eval(f))

