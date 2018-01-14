class Animal:
    def __init__(self):
        self.name = 'animal'
        self.__privateName = 'privateName'
        self._nickname = 'animal nick'    # 私有属性不会被继承
        print('animal init')

    def eat(self):
        print(self.name)
        print('animal eat ======')

    def drink(self):
        print('animal drink')

    def sleep(self):
        print('animal sleep')

    def run(self):
        print('animal run')

    def __kill(self):         # 私有方法不会被继承
        print('animal kill')


class Cat(Animal):
    def __init__(self, name=''):
        Animal.__init__(self)
        self.name = name
        print('cat init')

    def eat(self):
        print('cat eat')
        Animal.eat(self)        # 调用父类方法
        super().eat()
        # super().__kill()      # 父类私有方法不能被调用

    def catch(self):
        print('cat catch a mouse')
        print(self.name)
        print(self._nickname)


if __name__ == '__main__':
    cat = Cat('kitty')
    cat.eat()
    print(cat.name)
    cat.catch()

