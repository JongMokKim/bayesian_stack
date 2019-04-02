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

from DataSampler import index_gen
from utils.loss import UncertaintyLoss

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.args = args
        if args.uncertainty:
            self.fc2_var = nn.Linear(500,10)
            self.softp = nn.Softplus()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        if not self.args.uncertainty:
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        else:
            x_mu = self.fc2(x)
            x_var = self.fc2_var(x)
            x_var_pos = torch.exp(x_var)
            return x_mu, x_var_pos


def test(args, model, device, test_loader, data_idx, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    print('------start final prediction and save the result-----')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if args.uncertainty:
                output, output_var = model(data)
                test_loss += UncertaintyLoss(output, output_var, target, reduction='sum')
            else:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum')
            # output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.uncertainty:
                try:
                    predictions_mu = np.concatenate((predictions_mu,output.data.cpu().numpy()), 0)
                    predictions_var = np.concatenate((predictions_var,output_var.data.cpu().numpy()), 0)
                    labels = np.concatenate((labels,target.data.cpu().numpy()), 0)
                except:
                    predictions_mu = output.data.cpu().numpy()
                    predictions_var = output_var.data.cpu().numpy()
                    labels = target.data.cpu().numpy()

            else:
                try:
                    predictions = np.concatenate((predictions,output.data.cpu().numpy()), 0)
                    labels = np.concatenate((labels,target.data.cpu().numpy()), 0)
                except:
                    predictions = output.data.cpu().numpy()
                    labels = target.data.cpu().numpy()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_idx),
        100. * correct / len(data_idx)))
    if args.uncertainty:
        return predictions_mu, predictions_var, labels
    else:
        return predictions, labels


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--k-fold', type=int, default=6,
                        help='How many folds')
    parser.add_argument('--base-prediction', type=bool, default=True,
                        help='save base learner prediction ')
    parser.add_argument('--uncertainty', type=bool, default=True,
                        help='predict mu and var ? ')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


    train_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                            ])

    trainset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)

    train_idxs, val_idxs = index_gen(trainset, seed = 100, k=args.k_fold)

    ## restart in kfold - 3~4
    for fold in range(args.k_fold):
        print('Start {}-th fold Inference'.format(fold))

        val_loader = torch.utils.data.DataLoader(trainset,
            batch_size=args.test_batch_size, sampler=SubsetRandomSampler(val_idxs[fold]) , shuffle=False, **kwargs)

        model = Net(args).to(device)
        model.load_state_dict(torch.load('./run/uncertainty/[{}-th fold]mnist_cnn.pt'.format(fold)))

        if args.uncertainty:
            predictions_mu, predictions_var, labels = test(args, model, device, val_loader, val_idxs[fold], 1)
            try:
                predictions_mu_folds = np.concatenate((predictions_mu_folds, predictions_mu), 0)
                predictions_var_folds = np.concatenate((predictions_var_folds, predictions_var), 0)
                labels_folds = np.concatenate((labels_folds, labels), 0)
            except:
                predictions_mu_folds = predictions_mu
                predictions_var_folds = predictions_var
                labels_folds = labels
        else:
            predictions, labels = test(args, model, device, val_loader, val_idxs[fold], 1)
            try:
                predictions_folds = np.concatenate((predictions_folds, predictions), 0)
                labels_folds = np.concatenate((labels_folds, labels), 0)
            except:
                predictions_folds = predictions
                labels_folds = labels

        if args.base_prediction and fold == args.k_fold-1:
            if args.uncertainty:
                np.save('./run/uncertainty/x_mu.npy',predictions_mu_folds)
                np.save('./run/uncertainty/x_var.npy',predictions_var_folds)
                np.save('./run/uncertainty/y.npy',labels_folds)

            else:
                np.save('./run/baseline/x.npy',predictions_folds)
                np.save('./run/baseline/y.npy',labels_folds)


if __name__ == '__main__':
    main()