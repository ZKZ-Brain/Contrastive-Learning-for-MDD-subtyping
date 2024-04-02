"""Disentangle function related to HSIC"""

import torch
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
sigma=0.5

def sigma_estimation(Z, y):
    Z=Z.to(device)
    y=y.to(device)

    dim = 2
    p = 2
    dist_matrix = torch.norm(Z[:, None]-Z, dim,p)
    sigma = torch.mean(torch.sort(dist_matrix[:, :10], 1)[0])
    if sigma < 0.1:
        sigma = 0.1
    return sigma

def distmat(X):
    X=X.to(device)
    """ distance matrix"""
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)
    return D

def kernelmat(X, sigma, k_type="gaussian"):
    X=X.to(device)

    """ kernel matrix baker"""
    m = int(X.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])

    if k_type == "gaussian":
        Dxx = distmat(X)
        
        if sigma:
            variance = 2.*sigma*sigma*X.size()[1]            
            Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor)   # kernel matrices        
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X,X)
                Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))

    ## Adding linear kernel
    elif k_type == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)
    Kxc = torch.mm(Kx,H)

    return Kxc

def hsic_regular(x, y, sigma=sigma):
    x=x.to(device)
    y=y.to(device)

    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x, y, sigma=sigma):
    x=x.to(device)
    y=y.to(device)

    Pxy = hsic_regular(x, y, sigma)
    Px = torch.sqrt(hsic_regular(x, x, sigma))
    Py = torch.sqrt(hsic_regular(y, y, sigma))
    thehsic = Pxy/(Px*Py)
    return thehsic