
#%%
import numpy as np
import csv
import matplotlib.pyplot as plt


#%%
num_epochs = 40


#%%
fig = plt.figure()
for phase in ['train', 'valid']:

    list_epoch    = []
    list_loss     = []

    for epoch in range(num_epochs):
        with open('./log/{}_log_epoch{}.txt'.format(phase, epoch), 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            d = list(reader)
            list_epoch.append(float(d[0][0]))
            list_loss.append(float(d[0][2]))
        
    if phase == 'train':
        plt.plot(list_epoch, list_loss, color = 'red')
    else:
        plt.plot(list_epoch, list_loss, color = 'blue')

    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel('Loss', fontsize = 15)
    plt.ylim(0, 0.01)
    
plt.savefig('./log/loss.jpg', dpi = fig.dpi)


#%%
for phase in ['train', 'valid']:

    list_epoch    = []
    list_accuracy = []

    for epoch in range(num_epochs):
        with open('./log/{}_log_epoch{}.txt'.format(phase, epoch), 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            d = list(reader)
            list_epoch.append(float(d[0][0]))
            list_accuracy.append(float(d[0][1]))
        
    if phase == 'train':
        plt.plot(list_epoch, list_accuracy, color = 'red', label = 'Train: VGGFace2')
    else:
        plt.plot(list_epoch, list_accuracy, color = 'blue', label = 'Valid: LFW')

    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.ylim(0.55, 1)
    
plt.legend(loc='upper left')
plt.savefig('./log/accuracy.jpg', dpi = fig.dpi)


