#!/usr/bin/python
#-*- coding: utf-8 -*-

import scipy as sp
import matplotlib.pyplot as plt

#方差
def error(f, x, y):
	return sp.sum((f(x)-y)**2)

#提取数据
data = sp.genfromtxt("data/web_traffic.tsv", delimiter = "\t")
x = data[:, 0]
y = data[:, 1]
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

#1阶整体预测
fp1 = sp.polyfit(x, y, 1)
#fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full = True)
f1 = sp.poly1d(fp1)
print("error d=1: %s" %error(f1, x, y))

#分段拟合
#计算拐点的小时数
inflection = 3.5*7*24
xa = x[: inflection]
ya = y[: inflection]
xb = x[inflection :]
yb = y[inflection :]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
print("Error inflection = %f" %(fa_error + fb_error))

#散点图
plt.scatter(x, y, c = 'k', marker = 'o', s = 5)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight = True)

#生成x值用来作图
fx = sp.linspace(0, x[-1], 1000)

#1阶拟合图
plt.plot(fx, f1(fx), 'k', linewidth = 2)
#分段拟合图
fx1 = sp.linspace(0, x[-1], 1000)
plt.plot(fx1, fa(fx1), 'b--', linewidth = 2)
fx2 = sp.linspace(7*24*3.5, x[-1], 1000)
plt.plot(fx2, fb(fx2), 'r-.', linewidth = 2)


plt.legend(["d = %i" %f1.order], loc = 'upper left')
plt.grid()
plt.show()