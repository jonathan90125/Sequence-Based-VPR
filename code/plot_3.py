# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt

x = [3, 4, 5,6]
y1 = [0.89, 0.96, 0.98, 0.99]
y2 = [0.39, 0.49, 0.62, 0.72]
y3 = [0.90, 0.92, 0.92, 0.90]
y4 = [0.03, 0.10, 0.33, 0.38]
plt.plot(x, y1, color='blue', marker='.', ms=5, label="Gardens Point Dataset (day-to-day)")
plt.plot(x, y2, color='red', marker='.', ms=5, label="Gardens Point Dataset (day-to-night)")
plt.plot(x, y3, color='green', marker='.', ms=5, label="Campus Loop Dataset")
plt.plot(x, y4, color='orange', marker='.', ms=5, label="NordLand Dataset")

y1 = [0.92, 0.93, 0.94, 0.95]
y2 = [0.41, 0.42, 0.42, 0.43]
y3 = [0.90, 0.91, 0.92, 0.92]
y4 = [0.22, 0.24, 0.25, 0.26]
plt.plot(x, y1, color='blue', marker='.', ms=5, linestyle='dashed', label="Gardens Point Dataset (day-to-day)")
plt.plot(x, y2, color='red', marker='.', ms=5, linestyle='dashed', label="Gardens Point Dataset (day-to-night)")
plt.plot(x, y3, color='green', marker='.', ms=5, linestyle='dashed', label="Campus Loop Dataset")
plt.plot(x, y4, color='orange', marker='.', ms=5, linestyle='dashed', label="NordLand Dataset")

plt.xticks(rotation=45)
plt.xlabel("length of query list")
plt.ylabel("accuracy")
plt.title("Accuracy of various sequence length")

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8)
plt.tight_layout()
plt.savefig("a.jpg")
plt.show()
