
class Person(object):
    number = 61

    def __init__(self):
        self.name = '小明'
        self.age = 18
        self.gender = '男'

    def func(self):
        pass


class Student(Person):
    def eat(self):
        print('chi')

print(Student.mro)
stu1 = Student()
print(stu1.name, stu1.func(), stu1.age, stu1.eat(), stu1.gender, stu1.number)