
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms


def index_gen(dataset, seed, k):

    X = dataset.train_data
    y = dataset.train_labels

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state = seed)
    skf.get_n_splits(X, y)

    train_idxs = []
    val_idxs= []

    for i, (train_idx, val_idx) in enumerate(skf.split(X,y)):
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)
        print('{}-th fold trian/val idxs'.format(i))
        print(train_idx)
        print(val_idx)

    return train_idxs, val_idxs


if __name__ == '__main__':


    dataset =  datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    mask_gen(dataset, seed=100)



