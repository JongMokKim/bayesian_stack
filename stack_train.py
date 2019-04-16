from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data.sampler import Sampler, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorboardX
from scipy.special import softmax

from DataSampler import index_gen
from utils.loss import UncertaintyLoss
from utils.custom_dataloader import MyDataset


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(44,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,10)

        self.vfc1 = nn.Linear(4,8)
        self.vfc2 = nn.Linear(8,4)

        # self.fc1 = nn.Linear(44, 22)
        # self.fc2 = nn.Linear(22, 10)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 10)
        self.args = args
        self.init_weight()

        self.dropout = nn.Dropout(0.5)

    def init_weight(self):
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.vfc1.bias.data.fill_(0)
        self.vfc2.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.vfc1.weight)
        nn.init.xavier_uniform_(self.vfc2.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # self.fc1.bias.data.fill_(0)
        # self.fc2.bias.data.fill_(0)
        # self.fc3.bias.data.fill_(0)
        # self.fc4.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # nn.init.xavier_uniform_(self.fc4.weight)


    def forward(self, x):
        x_mu, x_var = x[:,:40], x[:,40:]

        x_var_mul = self.vfc1(x_var)
        x_var_mul = self.vfc2(x_var_mul)

        # x_mu = x_mu.view(-1, 4, 10)
        # x_mu = x_mu * x_var_mul[...,np.newaxis]
        # x_mu = x_mu.view(-1,40)

        # x = torch.cat((x_mu,x_var), dim=1)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.fc3(x)

        # x = (self.fc1(x))
        # x = (self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return x


def train(args, model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss=loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 12000,
                       100. * batch_idx * len(data) // 12000, loss.item()))


def test(args, model, device, test_loader, epoch, loss_function):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss=loss_function(output,target)
            test_loss += loss.item()
            # output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= 10000//args.test_batch_size

    print('\nTest set [Epoch:{}]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch,
        test_loss, correct, 10000,
        100. * correct / 10000))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--uncertainty', type=bool, default=True,
                        help='predict mu and var ? ')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data-mode', type=str,  default='softmax',
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    load_folder = 'k-fold_learning'

    ## Load Data from base-learner train
    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    if args.data_mode == 'onehot':
        train_x_mu_0 = one_hot(np.argmax(np.load('./run/{}/train_x_mu_fold_0.npy'.format(load_folder)), axis=1),10)
        train_x_mu_1 = one_hot(np.argmax(np.load('./run/{}/train_x_mu_fold_1.npy'.format(load_folder)), axis=1),10)
        train_x_mu_2 = one_hot(np.argmax(np.load('./run/{}/train_x_mu_fold_2.npy'.format(load_folder)), axis=1),10)
        train_x_mu_3 = one_hot(np.argmax(np.load('./run/{}/train_x_mu_fold_3.npy'.format(load_folder)), axis=1),10)

        test_x_mu_0 = one_hot(np.argmax(np.load('./run/{}/test_x_mu_fold_0.npy'.format(load_folder)), axis=1),10)
        test_x_mu_1 = one_hot(np.argmax(np.load('./run/{}/test_x_mu_fold_1.npy'.format(load_folder)), axis=1),10)
        test_x_mu_2 = one_hot(np.argmax(np.load('./run/{}/test_x_mu_fold_2.npy'.format(load_folder)), axis=1),10)
        test_x_mu_3 = one_hot(np.argmax(np.load('./run/{}/test_x_mu_fold_3.npy'.format(load_folder)), axis=1),10)

    elif args.data_mode == 'softmax':
        train_x_mu_0 = softmax(np.load('./run/{}/train_x_mu_fold_0.npy'.format(load_folder)), axis=1)
        train_x_mu_1 = softmax(np.load('./run/{}/train_x_mu_fold_1.npy'.format(load_folder)), axis=1)
        train_x_mu_2 = softmax(np.load('./run/{}/train_x_mu_fold_2.npy'.format(load_folder)), axis=1)
        train_x_mu_3 = softmax(np.load('./run/{}/train_x_mu_fold_3.npy'.format(load_folder)), axis=1)

        test_x_mu_0 = softmax(np.load('./run/{}/test_x_mu_fold_0.npy'.format(load_folder)), axis=1)
        test_x_mu_1 = softmax(np.load('./run/{}/test_x_mu_fold_1.npy'.format(load_folder)), axis=1)
        test_x_mu_2 = softmax(np.load('./run/{}/test_x_mu_fold_2.npy'.format(load_folder)), axis=1)
        test_x_mu_3 = softmax(np.load('./run/{}/test_x_mu_fold_3.npy'.format(load_folder)), axis=1)

    train_x_var_0 = np.load('./run/{}/train_x_var_fold_0.npy'.format(load_folder))
    train_x_var_1 = np.load('./run/{}/train_x_var_fold_1.npy'.format(load_folder))
    train_x_var_2 = np.load('./run/{}/train_x_var_fold_2.npy'.format(load_folder))
    train_x_var_3 = np.load('./run/{}/train_x_var_fold_3.npy'.format(load_folder))

    train_y = np.load('./run/{}/train_y.npy'.format(load_folder))

    test_x_var_0 = np.load('./run/{}/test_x_var_fold_0.npy'.format(load_folder))
    test_x_var_1 = np.load('./run/{}/test_x_var_fold_1.npy'.format(load_folder))
    test_x_var_2 = np.load('./run/{}/test_x_var_fold_2.npy'.format(load_folder))
    test_x_var_3 = np.load('./run/{}/test_x_var_fold_3.npy'.format(load_folder))

    test_y = np.load('./run/{}/test_y.npy'.format(load_folder))

    train_x = np.concatenate((train_x_mu_0, train_x_mu_1, train_x_mu_2,  train_x_mu_3
                              ,train_x_var_0,train_x_var_1,train_x_var_2,train_x_var_3
                              ), axis=1)
    test_x = np.concatenate((test_x_mu_0, test_x_mu_1, test_x_mu_2,  test_x_mu_3
                              ,test_x_var_0,test_x_var_1,test_x_var_2,test_x_var_3
                              ), axis=1)



    trainset = MyDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)

    valset = MyDataset(test_x, test_y)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)

    model = Net(args).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):

        loss_function = nn.CrossEntropyLoss(reduction='mean')
        train(args, model, device, train_loader, optimizer, epoch, loss_function)
        test(args, model, device, val_loader, epoch, loss_function)

        torch.save(model.state_dict(), "./run/kfold_metalearner/model.pt")



if __name__ == '__main__':
    main()