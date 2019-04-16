from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import PIL

from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedKFold


from DataSampler import index_gen
from utils.loss import UncertaintyLoss, SoftRegressLoss, RegressionLoss, MDN_ClassifyLoss

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        # basenet
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        # multi-head
        # phi0~2, mu0~2, sigma0~2
        self.p0_fc1 = nn.Linear(4 * 4 * 50, 500)
        self.p0_fc2 = nn.Linear(500, 1)
        self.p1_fc1 = nn.Linear(4 * 4 * 50, 500)
        self.p1_fc2 = nn.Linear(500, 1)
        self.p2_fc1 = nn.Linear(4 * 4 * 50, 500)
        self.p2_fc2 = nn.Linear(500, 1)

        self.m0_fc1 = nn.Linear(4*4*50, 500)
        self.m0_fc2 = nn.Linear(500,10)
        self.m1_fc1 = nn.Linear(4*4*50, 500)
        self.m1_fc2 = nn.Linear(500,10)
        self.m2_fc1 = nn.Linear(4*4*50, 500)
        self.m2_fc2 = nn.Linear(500,10)

        self.s0_fc1 = nn.Linear(4*4*50, 500)
        self.s0_fc2 = nn.Linear(500,10)
        self.s1_fc1 = nn.Linear(4*4*50, 500)
        self.s1_fc2 = nn.Linear(500,10)
        self.s2_fc1 = nn.Linear(4*4*50, 500)
        self.s2_fc2 = nn.Linear(500,10)

        self.softmax = nn.Softmax(dim=1)
        self.sigma_max = 5
        self.sigmoid = nn.Sigmoid()
        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        ## multi head!!
        m0 = F.relu(self.m0_fc1(x))
        m0 = self.m0_fc2(m0)
        m1 = F.relu(self.m1_fc1(x))
        m1 = self.m1_fc2(m1)
        m2 = F.relu(self.m2_fc1(x))
        m2 = self.m2_fc2(m2)

        s0 = F.relu(self.s0_fc1(x))
        s0 = self.s0_fc2(s0)
        s1 = F.relu(self.s1_fc1(x))
        s1 = self.s1_fc2(s1)
        s2 = F.relu(self.s2_fc1(x))
        s2 = self.s2_fc2(s2)

        p0 = F.relu(self.p0_fc1(x))
        p0 = self.p0_fc2(p0)
        p1 = F.relu(self.p1_fc1(x))
        p1 = self.p1_fc2(p1)
        p2 = F.relu(self.p2_fc1(x))
        p2 = self.p2_fc2(p2)

        ## normalize phi
        max_p = torch.max(torch.cat((p0,p1,p2),dim=1),dim=1)[0][...,np.newaxis]
        p0 = p0-max_p
        p1 = p1-max_p
        p2 = p2-max_p
        phi_vec = self.softmax(torch.cat((p0,p1,p2),dim=1))

        ## normalize sigma
        s0 = self.sigmoid(s0) * self.sigma_max
        s1 = self.sigmoid(s1) * self.sigma_max
        s2 = self.sigmoid(s2) * self.sigma_max
        m_vec = torch.cat
        return torch.stack((m0, m1, m2),dim=2), torch.stack((s0,s1,s2),dim=2), phi_vec


def train(args, model, device, train_loader, optimizer, epoch, data_idx):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        # if args.flip:
        #     data = -1. * data
        optimizer.zero_grad()

        mu_list, sigma_list, phi_vec = model(data)
        loss = MDN_ClassifyLoss(mu_list, sigma_list, phi_vec, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_idx),
                       100. * batch_idx * len(data) // len(data_idx), loss.item()))


def test(args, model, device, test_loader, data_idx, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            ori_target = target

            data, target = data.to(device), target.to(device)
            # if args.flip:
            #     data = -1. * data
            mu_vec, sigma_vec, phi_vec = model(data)
            test_loss += MDN_ClassifyLoss(mu_vec, sigma_vec, phi_vec, target)
            output= F.softmax(mu_vec, dim=1) * phi_vec.view(-1,1,3)
            output = output.sum(dim=2)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(ori_target[...,np.newaxis].cuda()).sum().item()

            try:
                predictions_mu = np.concatenate((predictions_mu,mu_vec.data.cpu().numpy()), 0)
                predictions_sigma = np.concatenate((predictions_sigma, sigma_vec.data.cpu().numpy()), 0)
                predictions_phi = np.concatenate((predictions_phi, phi_vec.data.cpu().numpy()), 0)
                labels = np.concatenate((labels,target.data.cpu().numpy()), 0)
            except:
                predictions_mu = mu_vec.data.cpu().numpy()
                predictions_sigma = sigma_vec.data.cpu().numpy()
                predictions_phi = phi_vec.data.cpu().numpy()
                labels = target.data.cpu().numpy()

    test_loss /= len(test_loader)
    test_loss *= args.test_batch_size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader)*args.test_batch_size,
        100. * correct / len(test_loader)))

    if epoch == args.epochs:
        print('------return the final prediction-----')
        return predictions_mu, predictions_sigma, predictions_phi, labels


def main(fold=0):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.04, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--k-fold', type=int, default=5,
                        help='How many folds')



    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.trial_name = 'mdn_init'

    if not os.path.isdir('run/{}'.format(args.trial_name)):
        os.mkdir('run/{}'.format(args.trial_name))

    # args.rotate = rotate
    # args.flip = flip

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    train_transform = transforms.Compose([
                            # transforms.ToPILImage(),
                            # transforms.RandomRotation((args.rotate,args.rotate)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                            ])

    trainset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    testset = datasets.MNIST('../data', train=False, download=True, transform=train_transform)

    train_idxs, val_idxs = index_gen(trainset, seed = 100, k=args.k_fold)


    print('Start {}-SPlit train'.format(fold))
    train_loader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, sampler=SubsetRandomSampler(val_idxs[fold]) , shuffle=False, **kwargs)

    val_loader = torch.utils.data.DataLoader(trainset,
        batch_size=args.test_batch_size, sampler=SequentialSampler(val_idxs[4]) , shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(testset,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net(args).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch, val_idxs[fold])
        if epoch == args.epochs :
            predictions_mu, predictions_sigma, predictions_phi, labels = test(args, model, device, val_loader, val_idxs[fold], epoch)
            np.save('./run/{}/train_x_mu_fold_{}.npy'.format(args.trial_name,fold),predictions_mu)
            np.save('./run/{}/train_x_sigma_fold_{}.npy'.format(args.trial_name,fold),predictions_sigma)
            np.save('./run/{}/train_x_phi_fold_{}.npy'.format(args.trial_name,fold),predictions_phi)
            np.save('./run/{}/train_y.npy'.format(args.trial_name),labels)

            predictions_mu, predictions_sigma, predictions_phi, labels = test(args, model, device, test_loader, val_idxs[fold], epoch)
            np.save('./run/{}/test_x_mu_fold_{}.npy'.format(args.trial_name,fold),predictions_mu)
            np.save('./run/{}/test_x_sigma_fold_{}.npy'.format(args.trial_name,fold),predictions_sigma)
            np.save('./run/{}/test_x_phi_fold_{}.npy'.format(args.trial_name,fold),predictions_phi)
            np.save('./run/{}/test_y.npy'.format(args.trial_name),labels)

        else:
            test(args, model, device, val_loader, val_idxs[fold], epoch)



    torch.save(model.state_dict(), "./run/{}/[{}-fold]mnist_cnn.pt".format(args.trial_name,fold))


if __name__ == '__main__':
    main(fold=0)
    main(fold=1)
    main(fold=2)
    main(fold=3)