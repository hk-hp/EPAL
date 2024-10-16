import json
from matplotlib import pyplot as plt
import numpy as np
import torch

loss = torch.load('loss.pt')
num = 84
kind = ['all_event', 'single_event', 'multi_event']
fig, axes = plt.subplots(2, 2, sharey=True)
ax = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
for j in range(3):
    x = np.arange(1, num, 1)
    y = np.zeros([3, num - 1])
    label = ['precision', 'recall', 'f1']
    for i in range(3,num):
        epoch_formatted = str(i)
        epoch_formatted = '0' * (3 - len(epoch_formatted)) + epoch_formatted
        with open('Result/exp_ch3/exp_ch3_'+ epoch_formatted + '.json', encoding='utf-8') as f:
            data = json.load(f)['test']['event'][kind[j]]
            y[0, i - 1] = data['micro_precision']
            y[1, i - 1] = data['micro_recall']
            y[2, i - 1] = data['micro_f1']

    ax[j].set_title(kind[j])
    for i in range(3):
        ax[j].plot(x, y[i])
    
    print(kind[j], np.argmax(y[2]) + 1, np.max(y[2]))
    ax[j].legend(label)

fig.suptitle('exp_2')
plt.savefig('res.jpg')

fig, axes = plt.subplots(1, 1, sharey=True)
x = np.arange(1, len(loss) + 1, 1)
y = np.array(loss)
label = ['precision', 'recall', 'f1']
# ax[3].set_ylim(0, max(y))
axes.plot(x, y)
plt.savefig('loss.jpg')

