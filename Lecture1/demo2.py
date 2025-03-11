import turtle
import time
import random

# 设置屏幕
wn = turtle.Screen()
wn.title("贪吃蛇游戏")
wn.bgcolor("green")
wn.setup(width=600, height=600)
wn.tracer(0)  # 关闭屏幕更新

# 蛇头
head = turtle.Turtle()
head.speed(0)
head.shape("square")
head.color("black")
head.penup()
head.goto(0, 0)
head.direction = "stop"

# 食物
food = turtle.Turtle()
food.speed(0)
food.shape("circle")
food.color("red")
food.penup()
x = random.randint(-290, 290)
y = random.randint(-290, 290)
food.goto(x, y)

segments = []

# 函数
def go_up():
    if head.direction != "down":
        head.direction = "up"


def go_down():
    if head.direction != "up":
        head.direction = "down"


def go_left():
    if head.direction != "right":
        head.direction = "left"


def go_right():
    if head.direction != "left":
        head.direction = "right"


def move():
    if head.direction == "up":
        y = head.ycor()
        head.sety(y + 20)
    elif head.direction == "down":
        y = head.ycor()
        head.sety(y - 20)
    elif head.direction == "left":
        x = head.xcor()
        head.setx(x - 20)
    elif head.direction == "right":
        x = head.xcor()
        head.setx(x + 20)


# 键盘绑定
wn.listen()
wn.onkeypress(go_up, "w")
wn.onkeypress(go_down, "s")
wn.onkeypress(go_left, "a")
wn.onkeypress(go_right, "d")

# 主游戏循环
while True:
    wn.update()

    # 检查是否撞到边界
    if head.xcor() > 290 or head.xcor() < -290 or head.ycor() > 290 or head.ycor() < -290:
        time.sleep(1)
        head.goto(0, 0)
        head.direction = "stop"

        # 隐藏蛇的身体部分
        for segment in segments:
            segment.goto(1000, 1000)

        # 清空身体部分列表
        segments.clear()

    # 检查是否吃到食物
    if head.distance(food) < 20:
        # 移动食物到随机位置
        x = random.randint(-290, 290)
        y = random.randint(-290, 290)
        food.goto(x, y)

        # 添加新的身体部分
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("square")
        new_segment.color("grey")
        new_segment.penup()
        segments.append(new_segment)

    # 移动蛇的身体部分，从最后一个部分开始移动到前一个部分的位置
    for index in range(len(segments) - 1, 0, -1):
        x = segments[index - 1].xcor()
        y = segments[index - 1].ycor()
        segments[index].goto(x, y)

    # 移动第一个身体部分到蛇头的位置
    if len(segments) > 0:
        x = head.xcor()
        y = head.ycor()
        segments[0].goto(x, y)

    move()

    # 检查是否撞到自己的身体
    for segment in segments:
        if segment.distance(head) < 20:
            time.sleep(1)
            head.goto(0, 0)
            head.direction = "stop"

            # 隐藏蛇的身体部分
            for seg in segments:
                seg.goto(1000, 1000)

            # 清空身体部分列表
            segments.clear()

    time.sleep(0.1)
    