import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# from nn import odCNN

models = ['128*1', '64*2', '42*3', '32*4']
params = [4611, 6467, 5127, 4323]
colors = ['#999900'] * 4

models.extend(['8@3', '8@5', '8@7', '8@11', '8@13', '8@15', '8@16'])
params.extend([11835, 10379, 8923, 6011, 4555, 3099, 2371])
colors.extend(['#0077FF'] * 7)

models.extend(['16@3', '16@5', '16@7', '16@11', '16@13', '16@15', '16@16'])
params.extend([23651, 20995, 18339, 13027, 10371, 7715, 6387])
colors.extend(['C9'] * 7)

models.extend(['32@3', '32@5', '32@7', '32@11', '32@13', '32@15', '32@16'])
params.extend([48435, 44147, 39859, 31283, 26995, 22707, 20563])
colors.extend(['#0077FF'] * 7)

models.extend(['64@3', '64@5', '64@7', '64@11', '64@13', '64@15', '64@16'])
params.extend([102611, 98131, 93651, 84691, 80211, 75731, 73491])
colors.extend(['C9'] * 7)

# models.extend(['96@3', '96@5', '96@7', '96@11', '96@13', '96@15', '96@16'])
# params.extend([162931, 162355, 161779, 160627, 160051, 159475, 159187])

models.extend(['128@3', '128@5', '128@7', '128@11', '128@13', '128@15', '128@16'])
params.extend([229395, 236819, 244243, 259091, 266515, 273939, 277651])
colors.extend(['#0077FF'] * 7)

models.extend(['256@3', '256@5', '256@7', '256@11', '256@13', '256@15', '256@16'])
params.extend([556691, 637075, 717459, 878227, 958611, 1038995, 1079187])
colors.extend(['C9'] * 7)

models.append('Multi-headed')
params.append(467155)
colors.append('C3')

# myNet = odCNN(n_filters=96, k_size=16)
# exit()

fig = plt.figure(figsize=(20.69, 4), dpi=120)
plt.bar(x=models, height=params, color=colors)
plt.xticks(rotation=50, fontsize=14)
ax = plt.gca()
for i, tick_label in enumerate(ax.get_xticklabels()):
    if i <= 3:
        tick_label.set_color('#999900')
    elif 4 <= i <= 10 or 18 <= i <= 24 or 32 <= i <= 38:
        tick_label.set_color('#0077FF')
    elif 11 <= i <= 17 or 25 <= i <= 31 or 39 <= i <= 45:
        tick_label.set_color('C9')
    else:
        tick_label.set_color('C3')

plt.text(1.5, 1000000, 'vNNs', color='#999900', fontsize=18, horizontalalignment='center')
plt.text(9, 1000000, '1D-CNNs', color='C0', fontsize=18)

plt.text(7, 800000, '8 filters', color='#0077FF', fontsize=14, horizontalalignment='center')
plt.text(14, 800000, '16 filters', color='C9', fontsize=14, horizontalalignment='center')
plt.text(21, 800000, '32 filters', color='#0077FF', fontsize=14, horizontalalignment='center')
plt.text(28, 800000, '64 filters', color='C9', fontsize=14, horizontalalignment='center')
plt.text(35, 800000, '128 filters', color='#0077FF', fontsize=14, horizontalalignment='center')
plt.text(40, 800000, '256 filters', color='C9', fontsize=14, horizontalalignment='center')

for x, y in zip(models, params):
    if x[-2] == '*':
        plt.text(x, y + 20000, '{:,}'.format(y), fontsize=14, ha='center', va='bottom', rotation=90, color='#999900')
    elif x[:2] == '8@' or x[:3] == '32@' or x[:4] == '128@':
        plt.text(x, y + 20000, '{:,}'.format(y), fontsize=14, ha='center', va='bottom', rotation=90, color='#0077FF')
    elif x[:3] == '16@' or x[:3] == '64@':
        plt.text(x, y + 20000, '{:,}'.format(y), fontsize=14, ha='center', va='bottom', rotation=90, color='C9')
    elif x[:4] == '256@':
        plt.text(x, y, '{:,}'.format(y), ha='center', fontsize=14, va='top', rotation=90, weight='bold', color='w')
    else:
        plt.text(x, y + 20000, '{:,}'.format(y), fontsize=14, ha='center', va='bottom', rotation=90, color='C3', weight='bold')

# x_bd_left, x_bd_right = plt.xlim()
y_bd_low, y_bd_high = plt.ylim()
h_rect = y_bd_high - y_bd_low
myRect_vnn = plt.Rectangle((-0.8, y_bd_low), 4.3, h_rect, color='#FFFF00', fill=True, lw=0.1, alpha=0.1)  # 近原点顶角坐标，宽，高
ax.add_patch(myRect_vnn)
myRect_1dcnn_8f = plt.Rectangle((3.5, y_bd_low), 7, h_rect, color='#0077FF', fill=True, lw=0.1, alpha=0.1)
ax.add_patch(myRect_1dcnn_8f)
myRect_1dcnn_16f = plt.Rectangle((10.5, y_bd_low), 7, h_rect, color='#00FFFF', fill=True, lw=0.1, alpha=0.1)
ax.add_patch(myRect_1dcnn_16f)
myRect_1dcnn_32f = plt.Rectangle((17.5, y_bd_low), 7, h_rect, color='#0077FF', fill=True, lw=0.1, alpha=0.1)
ax.add_patch(myRect_1dcnn_32f)
myRect_1dcnn_64f = plt.Rectangle((24.5, y_bd_low), 7, h_rect, color='#00FFFF', fill=True, lw=0.1, alpha=0.1)
ax.add_patch(myRect_1dcnn_64f)
myRect_1dcnn_128f = plt.Rectangle((31.5, y_bd_low), 7, h_rect, color='#0077FF', fill=True, lw=0.1, alpha=0.1)
ax.add_patch(myRect_1dcnn_128f)
myRect_1dcnn_256f = plt.Rectangle((38.5, y_bd_low), 7, h_rect, color='#00FFFF', fill=True, lw=0.1, alpha=0.1)
ax.add_patch(myRect_1dcnn_256f)
myRect_mH = plt.Rectangle((45.5, y_bd_low), 1.3, h_rect, color='C3', fill=True, lw=0.1, alpha=0.1)
ax.add_patch(myRect_mH)

plt.axis([-0.8, 46.6, y_bd_low, y_bd_high])

plt.xlabel('Network model configurations', fontsize=14, labelpad=-18)
plt.ylabel('Trainable parameters', fontsize=14, labelpad=-4)
plt.tick_params(axis='y', labelsize=14)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.title('Comparison of computation cost among all models', fontsize=16)

# plt.show()
plt.savefig('./params.png', bbox_inches='tight', pad_inches=0.01)
plt.close()




