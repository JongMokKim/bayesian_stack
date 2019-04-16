import numpy as np
import matplotlib.pyplot as plt


var0 = np.load('./run/k-fold_learning/test_x_var_fold_0.npy')
var90 = np.load('./run/k-fold_learning/test_x_var_fold_1.npy')
var180 = np.load('./run/k-fold_learning/test_x_var_fold_2.npy')
var270 = np.load('./run/k-fold_learning/test_x_var_fold_3.npy')

mu0 = np.load('./run/k-fold_learning/test_x_mu_fold_0.npy')
mu90 = np.load('./run/k-fold_learning/test_x_mu_fold_1.npy')
mu180 = np.load('./run/k-fold_learning/test_x_mu_fold_2.npy')
mu270 = np.load('./run/k-fold_learning/test_x_mu_fold_3.npy')

y = np.load('./run/k-fold_learning/test_y.npy')


n=100
start=np.random.randint(0,10000-n)

# hit=np.zeros((n))
# for idx, i in enumerate(range(start, start+n)):
#     hit[idx] = var0[i,y[i]]

plt.plot(var0[start:start+n].mean(axis=1), color='g')
plt.plot(var90[start:start+n].mean(axis=1), color='r')
plt.plot(var180[start:start+n].mean(axis=1), color='b')
plt.plot(var270[start:start+n].mean(axis=1), color='m')

plt.plot(np.where(y[start:start+n] == mu0[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'gv--')
plt.plot(np.where(y[start:start+n] == mu90[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'r^--')
plt.plot(np.where(y[start:start+n] == mu180[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'b<--')
plt.plot(np.where(y[start:start+n] == mu270[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'm>--')

plt.grid(True)
plt.show()


## oracle accuracy

oracle = (mu0.argmax(axis=1) == y).astype(np.uint8) + (mu90.argmax(axis=1) == y).astype(np.uint8) + (mu180.argmax(axis=1) == y).astype(np.uint8) + (mu270.argmax(axis=1) == y).astype(np.uint8)
oracle = np.clip(oracle, 0,2) / 2

print(np.mean(oracle))

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

mu0 = sigmoid(mu0) - var0
mu90 = sigmoid(mu90) - var90
# mu180 = sigmoid(mu180) - var180
mu270 = sigmoid(mu270) - var270
# oracle = (mu0.argmax(axis=1) == y.argmax(axis=1)) + (mu90.argmax(axis=1) == y.argmax(axis=1)) + (mu180.argmax(axis=1) == y.argmax(axis=1)) + (mu270.argmax(axis=1) == y.argmax(axis=1))
# oracle = np.clip(oracle, 0,1)

# print(np.mean(oracle))









