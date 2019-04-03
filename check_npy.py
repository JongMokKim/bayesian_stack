import numpy as np
import matplotlib.pyplot as plt

mu = np.load('./run/uncertainty_gaussian/x_mu.npy')
var = np.load('./run/uncertainty_gaussian/x_var.npy')
y = np.load('./run/uncertainty_gaussian/y.npy')

hit_std = 0
miss_std = 0
hit=0
miss=0

hit_stds=[]
miss_stds=[]

for i in range(60000):
    true = y[i]
    pred = np.argmax(mu[i])
    if pred == true:
        hit_std += np.sqrt(var[i][pred])
        hit+=1
        hit_stds.append(np.sqrt(var[i][pred]))
    else:
        miss_std += np.sqrt(var[i][pred])
        miss+=1
        miss_stds.append(np.sqrt(var[i][pred]))

    # print('[HIT CLASS]Mu:{}, Std:{}'.format(mu[i][cls],np.sqrt(np.exp(var[i][cls]))))
    # print('[OTHER CLASS]Mu:{}, Std:{}'.format(np.mean(mu[i])             , np.mean(np.sqrt(np.exp(var[i])))          )          )

hit_std /= hit
miss_std /= miss

print('hit std : {}   miss std : {}'.format(hit_std, miss_std))
print('hit:{}, miss:{}'.format(hit,miss))

n, bins, patches = plt.hist(hit_stds,100,density=True, facecolor='g', alpha=0.75)
plt.xlabel('std values')
plt.ylabel('probability')
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(miss_stds,100,density=True, facecolor='r', alpha=0.75)

plt.xlabel('std values')
plt.ylabel('probability')
plt.grid(True)
plt.show()











