import numpy as np


mu = np.load('./run/uncertainty/x_mu.npy')
var = np.load('./run/uncertainty/x_var.npy')
y = np.load('./run/uncertainty/y.npy')

hit_std = 0
miss_std = 0
hit=0
miss=0
for i in range(60000):
    true = y[i]
    pred = np.argmax(mu[i])
    if pred == true:
        hit_std += np.sqrt(var[i][pred])
        hit+=1
    else:
        miss_std += np.sqrt(var[i][pred])
        miss+=1

    # print('[HIT CLASS]Mu:{}, Std:{}'.format(mu[i][cls],np.sqrt(np.exp(var[i][cls]))))
    # print('[OTHER CLASS]Mu:{}, Std:{}'.format(np.mean(mu[i])             , np.mean(np.sqrt(np.exp(var[i])))          )          )

hit_std /= hit
miss_std /= miss

print('hit std : {}   miss std : {}'.format(hit_std, miss_std))
print('hit:{}, miss:{}'.format(hit,miss))

