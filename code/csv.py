
import matplotlib.pyplot as plt
import numpy as np


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ('Our Method, M=5', 'Our Method, M=6', 'CoHoG', 'ConvSequential-SLAM','NetVLAD', 'HOG','CALC','HybridNet','AMOSNet','SeqSLAM,M=5','SeqSLAM, M=10','RegionVLAD')
y_pos = np.arange(len(people))
performance1 = [0.92,0.9,0.69,1,0.98,0.21,0.29,0.69,0.6,0.36,0.44,0.55]
performance2 = [0.98,0.99,0.9,1,0.96,0.32,0.46,0.81,0.76,0.32,0.44,0.58]
performance3 = [0.63,0.72,0.36,0.61,0.57,0.21,0.18,0.45,0.48,0.03,0.05,0.46]
performance4 = [0.34,0.39,0.19,0.54,0.41,0.02,0.081,0.19,0.16,0.10,0.20,0.38]

error = np.random.rand(len(people))


total_width, n = 0.8, 2
width = total_width / n
y_pos=y_pos - (total_width - width) / 2

b=ax.barh(y_pos, performance1, align='center',
        color='seagreen', ecolor='black',height=0.2,label='Campus Loop Dataset')
#添加数据标签
for rect in b:
    w=rect.get_width()
    ax.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%w,ha='left',va='center')

b=ax.barh(y_pos+width/2, performance2, align='center',
        color='orangered', ecolor='black',height=0.2,label='Gardens Point Dataset (day-to-day)')
#添加数据标签
for rect in b:
    w=rect.get_width()
    ax.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%w,ha='left',va='center')

b=ax.barh(y_pos+2*width/2, performance3, align='center',
        color='skyblue', ecolor='black',height=0.2,label='Gardens Point Dataset (day-to-night)')
#添加数据标签
for rect in b:
    w=rect.get_width()
    ax.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%w,ha='left',va='center')

b=ax.barh(y_pos+3*width/2, performance4, align='center',
        color='sandybrown', ecolor='black',height=0.2,label='NordLand Dataset')
#添加数据标签
for rect in b:
    w=rect.get_width()
    ax.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%w,ha='left',va='center')


ax.set_yticks(y_pos+width/2.0)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy')
ax.set_title('Accuracy Comparison',fontsize=15)
plt.legend(loc='lower right', fontsize=10)

plt.savefig("accuracy.jpg")
plt.show()

