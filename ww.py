# coding=utf-8
# 测试utf-8编码
import sys

#reload(sys)
#sys.setdefaultencoding('utf-8')

list_a = []


def a():
    list_a = [1]  ## 语句1


a()
print
list_a  # []

print
"======================"

list_b = []


def b():
    list_b.append(1)  ## 语句2


b()
print
list_b  # [1]