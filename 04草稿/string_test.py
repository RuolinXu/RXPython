
name = 'my \tname is {name}, age is {age}.'

# 一共打印50个字符，把原始字符串放到中间，两边不够的用“-”补上
print(name.center(50, '-'))

intab = "aeiou"
outtab = "12345"
print(intab+outtab)                      # 字符拼接
print(name.translate(str.maketrans(intab, outtab)))  # 字符映射

print(name.find("name"))
print(name.format(age=26, name="aaron fan"))
print("Aaron FAn".lower())          # 把大写变成小写
print("Aaron FAn".upper())          # 把小写变成大写

print("aaron fan".replace("n","N",1))       # 替换字符串中的指定字符，这里的示例是替换其中一个n，使其变成N，值替换1个，也可以替换多个
print("aaron fan".rfind("n"))       # 从左网友数，找到最右边的那个值的下标

# 判断
print(name.endswith("an"))          # 判断一个字符串以什么结尾，比如如果以an结尾，就返回True，否则返回False
print(name.startswith('my'))        # 判断字符串是否以my开头
print("123".isdigit())              # 判断是否为整数，这个用的比较多
print("test123".isidentifier())     # 判断是否为一个合法的变量名
print("123".isnumeric())            # 判断是否只包含数字
print("Aaron Fan".istitle())        # 判断首字母是否全部为大写
print(name.isprintable())           # 判断这个东西是否可以打印，用到的时候再去详细查下吧
print("AARON FAN".isupper())        # 判断是否全部大写

print('sssssss {} ssssss {}llll{}'.format(1,2,3))