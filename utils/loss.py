import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace


def MDN_ClassifyLoss(mu_vec, sigma_vec, phi_vec, target, reduction='mean'):
    # batch_size = 64
    num_class = 10
    n_experts = 3
    n_sample = 10
    eps = 1E-10
    batch_size,_,_ = mu_vec.size()

    m = Laplace(torch.tensor([0.0]), torch.tensor([1.0])).expand([batch_size, num_class])
    expert_logit = torch.zeros((batch_size,num_class,n_experts)).cuda()
    for n in range(n_experts):
        for sample in range(n_sample):
            sample_logit=mu_vec[...,n] + sigma_vec[...,n]*m.sample().cuda()
            expert_logit[...,n] += F.softmax(sample_logit, dim=1)
        expert_logit[...,n] /= n_sample
    mean_logit = (phi_vec.view(-1,1,n_experts) * expert_logit).sum(dim=2)

    criterion = nn.NLLLoss(reduction=reduction).cuda()

    mean_logit = torch.log(mean_logit + eps)

    loss = criterion(mean_logit, target.long())
    if str(loss.data) == 'nan':
        print('nan!!!!!!!!!!!!!!!!!!!')
    return loss

def RegressionLoss(logit_mu, logit_var, target):

    n, _ = logit_mu.size()
    logit_mu = torch.nn.functional.sigmoid(logit_mu)

    logit_var = logit_var.clamp(min=1E-20)

    criterion = nn.MSELoss(reduction='none').cuda()
    loss = criterion(logit_mu, target)

    loss = loss/(2*logit_var) + (logit_var)/2
    # loss = loss.mean()
    # weight_mtx = (target*9 + 1)
    weight_mtx = 10

    loss = loss * weight_mtx

    return loss.mean()


# 1. calculate only true label
def SoftRegressLoss(logit_mu, logit_var, target, reduction='mean'):

    n, _ = logit_mu.size()
    logit_var = torch.squeeze(logit_var).clamp(min=1E-20)

    ## formal softmax-crossentropy loss
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    loss = criterion(logit_mu, target.long())

    ##regression-likely add variance term
    loss = loss/(2*logit_var) + (logit_var)/2
    loss = loss.mean()

    return loss



def UncertaintyLoss(logit_mu, logit_var, target, reduction='mean'):

    n, _ = logit_mu.size()
    m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    # eps = torch.randn(10)
    # sample_logit = logit_mu + torch.sqrt(logit_var)*eps[0]
    sample_logit = logit_mu + torch.sqrt(logit_var)*m.sample().cuda()
    mean_logit = F.softmax(sample_logit , dim=1)
    for i in range(1, 20):
        # sample_logit = logit_mu + torch.sqrt(logit_var)*eps[i]
        sample_logit = logit_mu + torch.sqrt(logit_var)*m.sample().cuda()
        mean_logit = mean_logit + F.softmax(sample_logit, dim=1)

    mean_logit /= 20
    criterion = nn.NLLLoss(reduction=reduction).cuda()

    mean_logit = torch.log(mean_logit)

    # loss = criterion(mu_log, target.long())
    loss = criterion(mean_logit, target.long())
    if str(loss.data) == 'nan':
        print(1)
    return loss

if __name__ =='__main__':
    MDN_ClassifyLoss(0,0,0)


