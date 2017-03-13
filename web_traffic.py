#!/usr/bin/python
#-*- coding: utf-8 -*-

import scipy as sp
import matplotlib.pyplot as plt

#求预测方差
def error(f, x, y):
	return sp.sum((f(x)-y)**2)

#提取数据
data = sp.genfromtxt("data/web_traffic.tsv", delimiter = "\t")
#print(data[: 10])

#时间
x = data[:, 0]
#访问量
y = data[:, 1]

#舍去错误数据，为空数据
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

#分训练&测试数据
inflection = 3.5*7*24
xa = x[: inflection]
ya = y[: inflection]
xb = x[inflection :]
yb = y[inflection :]

#1阶直线拟合
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full = True)
f1 = sp.poly1d(fp1)
f1b = sp.poly1d(sp.polyfit(xb, yb, 1))
print("error d=1: %s" %error(f1, x, y))
print("error da=1: %s" %error(f1b, xa, ya))

#阶数为2的多项式拟合
f2p = sp.polyfit(x, y, 2)
f2 = sp.poly1d(f2p)
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
print("error d=2: %s" %error(f2, x, y))
print("error da=2: %s" %error(fb2, xa, ya))

#阶数为3
f3p = sp.polyfit(x, y, 3)
f3 = sp.poly1d(f3p)
print("error d=3: %s" %error(f3, x, y))

#阶数为10
f10p = sp.polyfit(x, y, 10)
f10 = sp.poly1d(f10p)
print("error d=10: %s" %error(f10, x, y))

#阶数为100
#f100p = sp.polyfit(x, y, 100)
#f100 = sp.poly1d(f100p)
#print("error d=100: %s" %error(f100, x, y))


#实际值的散点图
plt.scatter(x, y, c = 'k', marker = 'o', s = 5, label = 'plot 1')
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight = True)

#生成x值用来作图
fx = sp.linspace(0, x[-1], 1000)  
#1阶预测直线
plt.plot(fx, f1(fx), 'k',linewidth = 2)
#2阶多项式预测图
plt.plot(fx, f2(fx), 'b--',linewidth = 2)
#3阶预测图
plt.plot(fx, f3(fx), 'g-.',linewidth = 3)
#10阶预测图
plt.plot(fx, f10(fx), 'c:',linewidth = 3)
#100阶预测图
#plt.plot(fx, f100(fx), 'r',linewidth = 2)

plt.legend(("d = %i" % f1.order, "d = %i" % f2.order, "d = %i" % f3.order, "d = %i" % f10.order), loc = "upper left")
plt.grid()
#plt.show()

#预测多久后会达到100000hits
print(f2)
print(fb2)
print(fb2-100000)
from scipy.optimize import fsolve
reached_max = fsolve(fb2-100000, 800)/(7*24)
print("100000 hits/hour expected at week %f" %reached_max[0])










