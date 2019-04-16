import numpy as np
import matplotlib.pyplot as plt

run_folder = 'mdn_init'

sigma0 = np.load('./run/{}/train_x_sigma_fold_0.npy'.format(run_folder))
sigma1 = np.load('./run/{}/train_x_sigma_fold_1.npy'.format(run_folder))
sigma2 = np.load('./run/{}/train_x_sigma_fold_2.npy'.format(run_folder))
sigma3 = np.load('./run/{}/train_x_sigma_fold_3.npy'.format(run_folder))

mu0 = np.load('./run/{}/train_x_mu_fold_0.npy'.format(run_folder))
mu1 = np.load('./run/{}/train_x_mu_fold_1.npy'.format(run_folder))
mu2 = np.load('./run/{}/train_x_mu_fold_2.npy'.format(run_folder))
mu3 = np.load('./run/{}/train_x_mu_fold_3.npy'.format(run_folder))

phi0 = np.load('./run/{}/train_x_phi_fold_0.npy'.format(run_folder))
phi1 = np.load('./run/{}/train_x_phi_fold_1.npy'.format(run_folder))
phi2 = np.load('./run/{}/train_x_phi_fold_2.npy'.format(run_folder))
phi3 = np.load('./run/{}/train_x_phi_fold_3.npy'.format(run_folder))

y = np.load('./run/{}/train_y.npy'.format(run_folder))

## variance for prediction element? or mean of vector ?
## First,mean of vector
bs,_ = phi0.shape

aleatoric0 = (sigma0 * phi0.reshape((bs,1,3))).sum(axis=2).mean(axis=1)
aleatoric1 = (sigma1 * phi1.reshape((bs,1,3))).sum(axis=2).mean(axis=1)
aleatoric2 = (sigma2 * phi2.reshape((bs,1,3))).sum(axis=2).mean(axis=1)
aleatoric3 = (sigma3 * phi3.reshape((bs,1,3))).sum(axis=2).mean(axis=1)

epistemic0 = (phi0.reshape((bs,1,3)) * np.square(np.abs(mu0 - (phi0.reshape((bs,1,3)) * mu0).sum(axis=2)[...,np.newaxis]))).sum(axis=2).mean(axis=1)
epistemic1 = (phi1.reshape((bs,1,3)) * np.square(np.abs(mu1 - (phi1.reshape((bs,1,3)) * mu1).sum(axis=2)[...,np.newaxis]))).sum(axis=2).mean(axis=1)
epistemic2 = (phi2.reshape((bs,1,3)) * np.square(np.abs(mu2 - (phi2.reshape((bs,1,3)) * mu2).sum(axis=2)[...,np.newaxis]))).sum(axis=2).mean(axis=1)
epistemic3 = (phi3.reshape((bs,1,3)) * np.square(np.abs(mu3 - (phi3.reshape((bs,1,3)) * mu3).sum(axis=2)[...,np.newaxis]))).sum(axis=2).mean(axis=1)


n=100
start=np.random.randint(0,10000-n)

# hit=np.zeros((n))
# for idx, i in enumerate(range(start, start+n)):
#     hit[idx] = var0[i,y[i]]

plt.plot(aleatoric0[start:start+n], color='g')
plt.plot(aleatoric1[start:start+n], color='r')
plt.plot(aleatoric2[start:start+n], color='b')
plt.plot(aleatoric3[start:start+n], color='m')

# plt.plot(np.where(y[start:start+n] == mu0[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'gv--')
# plt.plot(np.where(y[start:start+n] == mu90[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'r^--')
# plt.plot(np.where(y[start:start+n] == mu180[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'b<--')
# plt.plot(np.where(y[start:start+n] == mu270[start:start+n].argmax(axis=1),np.ones((n)),np.zeros((n))), 'm>--')

plt.grid(True)
plt.show()


# ## oracle accuracy
#
# oracle = (mu0.argmax(axis=1) == y).astype(np.uint8) + (mu90.argmax(axis=1) == y).astype(np.uint8) + (mu180.argmax(axis=1) == y).astype(np.uint8) + (mu270.argmax(axis=1) == y).astype(np.uint8)
# oracle = np.clip(oracle, 0,2) / 2
#
# print(np.mean(oracle))
#
# def sigmoid(x):
#     return np.exp(x)/(1+np.exp(x))
#
# mu0 = sigmoid(mu0) - var0
# mu90 = sigmoid(mu90) - var90
# # mu180 = sigmoid(mu180) - var180
# mu270 = sigmoid(mu270) - var270
# # oracle = (mu0.argmax(axis=1) == y.argmax(axis=1)) + (mu90.argmax(axis=1) == y.argmax(axis=1)) + (mu180.argmax(axis=1) == y.argmax(axis=1)) + (mu270.argmax(axis=1) == y.argmax(axis=1))
# # oracle = np.clip(oracle, 0,1)
#
# # print(np.mean(oracle))
#








