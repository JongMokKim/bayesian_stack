import torch
import torch.nn as nn
import torch.nn.functional as F

def UncertaintyLoss(logit_mu, logit_var, target, reduction='mean'):
    n, _ = logit_mu.size()

    eps = torch.randn(10)

    sample_logit = logit_mu + torch.sqrt(logit_var)*eps[0]
    mean_logit = F.softmax(sample_logit , dim=1)
    for i in range(1, eps.size()[0]):
        sample_logit = logit_mu + torch.sqrt(logit_var)*eps[i]
        mean_logit = mean_logit + F.softmax(sample_logit, dim=1)

    mean_logit /= eps.size()[0]

    criterion = nn.NLLLoss(reduction=reduction).cuda()

    mean_logit = torch.log(mean_logit.clamp(min=1e-10))
    loss = criterion(mean_logit, target.long())

    loss /= n

    return loss

