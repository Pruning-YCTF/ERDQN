import math
import random


class Array2D:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.data = []
        self.data = [[0 for y in range(h)] for x in range(w)]

    def showArray2D(self):
        for y in range(self.h):
            for x in range(self.w):
                print(self.data[x][y], end=' ')
            print("")

    def __getitem__(self, item):
        return self.data[item]



class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False

    def __str__(self):
        return "x:" + str(self.x) + ",y:" + str(self.y)


class RRT:
    def __init__(self, map2d, startPoint, endPoint):
        self.TreeNode = [] # 生长树所有结点
        self.randomPoint = Point(-1, -1) # 随机生成点
        self.path = []
        self.step_size = 1
        self.neighbourhood = 1
        self.TreeNode.append(startPoint)
        self.min_dis = 999
        self.startpoint = startPoint
        self.endpoint = endPoint
        self.father = {(startPoint.x, self.startpoint.y) : Point(-1, -1)}

    def Manhattan(self, Point1, Point2): # 计算曼哈顿距离
        return abs(Point1.x - Point2.x) + abs(Point1.y - Point2.y)

    def getExpandDir(self, point1, point2):
        n1 = (point2.x - point1.x, point2.y - point1.y)
        len = math.sqrt((point2.x - point1.x) * (point2.x - point1.x) + (point2.y - point1.y) * (point2.y - point1.y)) + 0.00001
        right = (1, 0)
        up = (0, 1)
        left = (-1, 0)
        down = (0, -1)
        cos_right = (n1[0] * right[0] + n1[1] * right[1]) / len
        cos_up = (n1[0] * up[0] + n1[1] * up[1]) / len
        cos_left = (n1[0] * left[0] + n1[1] * left[1]) / len
        cos_down = (n1[0] * down[0] + n1[1] * down[1]) / len
        cos_list = [(cos_right, 'right'), (cos_up, 'up'), (cos_left, 'left'), (cos_down, 'down')]
        min_cos = 1

        dir = ''
        for i in cos_list:
            if i[0] > 0 and i[0] <= min_cos:
                min_cos = i[0]
                dir = i[1]

        return dir

    def getRandomPoint(self): # 生成随机点
        x_range = map2d.w
        y_range = map2d.h
        x_new, y_new = 26, 14

        while map2d[x_new][y_new] == 1 or Point(x_new, y_new) in self.TreeNode:
            x_new = random.randint(0, x_range - 1)
            y_new = random.randint(0, y_range - 1)

        p = Point(x_new, y_new)
        return p

    def start(self):
        newPoint = self.startpoint
        while self.min_dis > self.neighbourhood:
            p = self.getRandomPoint()
            dis = self.Manhattan(p, self.startpoint)
            min_dis_point = self.startpoint

            for i in self.TreeNode:
                if self.Manhattan(i, p) < dis:
                    dis = self.Manhattan(i, p)
                    min_dis_point = i

            dir = self.getExpandDir(min_dis_point, p)
            if dir == 'right':
                if min_dis_point.x < 0 or min_dis_point.x >= map2d.w or min_dis_point.y < 0 or min_dis_point.y >= map2d.h or map2d[min_dis_point.x + self.step_size][min_dis_point.y] == 1:
                    continue
                else:
                    newPoint = Point(min_dis_point.x + self.step_size, min_dis_point.y)
                    self.TreeNode.append(newPoint)
                    self.father[(newPoint.x, newPoint.y)] = min_dis_point
            elif dir == 'up':
                if min_dis_point.x < 0 or min_dis_point.x >= map2d.w or min_dis_point.y < 0 or min_dis_point.y >= map2d.h or map2d[min_dis_point.x][min_dis_point.y + self.step_size] == 1:
                    continue
                else:
                    newPoint = Point(min_dis_point.x, min_dis_point.y + self.step_size)
                    self.TreeNode.append(newPoint)
                    self.father[(newPoint.x, newPoint.y)] = min_dis_point
            elif dir == 'down':
                if min_dis_point.x < 0 or min_dis_point.x >= map2d.w or min_dis_point.y < 0 or min_dis_point.y >= map2d.h or map2d[min_dis_point.x][min_dis_point.y - self.step_size] == 1:
                    continue
                else:
                    newPoint = Point(min_dis_point.x, min_dis_point.y - self.step_size)
                    self.TreeNode.append(newPoint)
                    self.father[(newPoint.x, newPoint.y)] = min_dis_point
            elif dir == 'left':
                if min_dis_point.x < 0 or min_dis_point.x >= map2d.w or min_dis_point.y < 0 or min_dis_point.y >= map2d.h or map2d[min_dis_point.x - self.step_size][min_dis_point.y] == 1:
                    continue
                else:
                    newPoint = Point(min_dis_point.x - self.step_size, min_dis_point.y)
                    self.TreeNode.append(newPoint)
                    self.father[(newPoint.x, newPoint.y)] = min_dis_point

            self.min_dis = self.Manhattan(newPoint, self.endpoint)

        self.TreeNode.append(self.endpoint)
        self.father[(self.endpoint.x, self.endpoint.y)] = newPoint

        i = (self.father[(self.endpoint.x, self.endpoint.y)].x, self.father[(self.endpoint.x, self.endpoint.y)].y)
        self.path.append(self.endpoint)
        self.path.append(Point(i[0], i[1]))
        print(len(self.path))
        while i != (-1, -1):
            self.path.append(self.father[i])
            i = (self.father[i].x, self.father[i].y)

        self.path.append(self.startpoint)
        return self.path

if __name__ == '__main__':
    map2d=Array2D(32,24)
    map2d[26][14] = 1
    map2d[28][12] = 1
    for i in range(3):
        map2d[i][10] = 1
        map2d[3][21 + i] = 1
        map2d[6 + i][18] = 1
        map2d[6 + i][19] = 1
        map2d[7][9 + i] = 1
        map2d[7 + i][12] = 1
        map2d[17 + i][7] = 1
        map2d[16 + i][21] = 1
        map2d[13 + i][18 + i] = 1
        map2d[19 + i][20 - i] = 1
        map2d[3][2 + i] = 1
        map2d[26 + i][12 + i] = 1
    for i in range(2):
        map2d[3][i] = 1
        map2d[4][i] = 1
        map2d[9 + i][2] = 1
        map2d[23 + i][2] = 1
        map2d[24][7 + i] = 1
    for i in range(4):
        map2d[13 + i][3] = 1
        map2d[16][4 + i] = 1
        map2d[28 + i][3] = 1
        map2d[28 + i][4] = 1
        map2d[27][20 + i] = 1

    #显示地图当前样子
    map2d.showArray2D()
    #创建RRT对象,并设置起点,终点
    aStar=RRT(map2d,Point(26,15),Point(31,23))
    #开始寻路
    pathList=aStar.start()
    #遍历路径点,在map2d上以'4'显示
    for point in pathList:
        map2d[point.x][point.y]=4
        # print(point)
    print("----------------------")
    #再次显示地图
    map2d.showArray2D()