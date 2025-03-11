import random
print("Hello World")

print("Hello World")

# 生成一个99乘法表函数
def multi_table():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print("{}*{}={}\t".format(j, i, i * j), end="")
        print("")


# 调用函数
multi_table()

# 生成随机数函数
print(random.randint(1, 100))
