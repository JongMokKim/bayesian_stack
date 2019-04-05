import numpy as np
import matplotlib.pyplot as plt

mu = np.load('./run/softmax_regression_softplus/x_mu.npy')
var = np.load('./run/softmax_regression_softplus/x_var.npy')
y = np.load('./run/softmax_regression_softplus/y.npy')

hit_std = 0
miss_std = 0
hit=0
miss=0

hit_stds=[]
miss_stds=[]

for i in range(60000):
    true = y[i]
    pred = np.argmax(mu[i])
    # pred=0
    if not str(var[i][0]) == 'nan':
        if pred == true:
            hit_std += (var[i][0])
            hit+=1
            hit_stds.append((var[i][0]))
        else:
            miss_std += (var[i][0])
            miss+=1
            miss_stds.append((var[i][0]))
    else:
        print(i)


hit_std /= hit
miss_std /= miss

print('hit std : {}   miss std : {}'.format(hit_std, miss_std))
print('hit:{}, miss:{}'.format(hit,miss))

n, bins, patches = plt.hist(hit_stds,100,density=True, facecolor='g', alpha=0.75)
plt.xlabel('std values')
plt.ylabel('probability')
plt.xlim((0,3))
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(miss_stds,100,density=True, facecolor='r', alpha=0.75)

plt.xlabel('std values')
plt.ylabel('probability')
plt.xlim((0,3))
plt.grid(True)
plt.show()












