# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt

x = [200, 400, 600, 800,1000]
y1 = [0.30, 0.52, 0.73, 0.97,1.31]
y2 = [0.25, 0.29, 0.34, 0.38,0.50]
y3 = [0.17, 0.24, 0.29, 0.32,0.41]
y4 = [0.14, 0.19, 0.26, 0.31,0.35]
plt.plot(x, y1, marker='.', ms=5, label="CoHoG")
plt.plot(x, y2, marker='.', ms=5, label="Our Method,SN=20")
plt.plot(x, y3, marker='.', ms=5, label="Our Method,SN=10")
plt.plot(x, y4, marker='.', ms=5, label="Our Method,SN=5")
plt.xticks(rotation=45)
plt.xlabel("length of database")
plt.ylabel("execution time(seconds)")
plt.title("Execution time Comparison")
plt.legend(loc="upper left",fontsize=8)
# 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式

plt.savefig("a.jpg")
plt.show()
