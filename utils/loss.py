import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace

# 1. calculate only true label
def SoftRegressLoss(logit_mu, logit_sigma, target, reduction='mean'):

    n, _ = logit_mu.size()
    logit_sigma = torch.squeeze(logit_sigma).clamp(min=1E-20)

    ## formal softmax-crossentropy loss
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    loss = criterion(logit_mu, target.long())

    ##regression-likely add variance term
    loss = loss/(2*logit_sigma) + (logit_sigma)/2
    loss = loss.mean()

    return loss



def UncertaintyLoss(logit_mu, logit_var, target, reduction='mean'):

    n, _ = logit_mu.size()
    m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    # eps = torch.randn(10)
    # sample_logit = logit_mu + torch.sqrt(logit_var)*eps[0]
    sample_logit = logit_mu + torch.sqrt(logit_var)*m.sample().cuda()
    mean_logit = F.softmax(sample_logit , dim=1)
    for i in range(1, 10):
        # sample_logit = logit_mu + torch.sqrt(logit_var)*eps[i]
        sample_logit = logit_mu + torch.sqrt(logit_var)*m.sample().cuda()
        mean_logit = mean_logit + F.softmax(sample_logit, dim=1)

    mean_logit /= 10
    # print('mean_logit',mean_logit)
    # print('logit_mu',F.softmax(logit_mu,dim=1))

    criterion = nn.NLLLoss(reduction=reduction).cuda()

    mean_logit = torch.log(mean_logit)
    # mu_log = torch.log(F.softmax(logit_mu))

    # loss = criterion(mu_log, target.long())
    loss = criterion(mean_logit, target.long())

    return loss


