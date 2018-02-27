import hashlib

hash = hashlib.sha256()
hash.update()



print(hs.sha256('123456'.encode('utf-8')).hexdigest())
hs1 = hs.sha256('123456'.encode('utf-8'))
hs1.update('123456'.encode('utf-8'))
print(hs1.hexdigest())
# 5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5
# 8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92

