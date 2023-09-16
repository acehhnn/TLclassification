import pandas as pd
import numpy as np
from numpy import linalg as LA

def emp_stieltjes_func(X, Y, lam):
    # n = 50
    # p = 100
    # Sigma = diag(c(rep(1,n), rep(0, p - n)))
    # lambda = 1
    
    M = ((sum(Y == 0) - 1) * np.cov(X[Y == 0,], rowvar = False) + (sum(Y == 1) - 1) * np.cov(X[Y == 1,], rowvar = False)) / (len(Y) - 2)
    eig_val, eig_vec = LA.eigh(M)
    xx = eig_val
    gamma = X.shape[1] / X.shape[0]
    
    m = np.mean(1 / (xx + lam))
    mp = np.mean(1 / pow((xx + lam), 2))
    v = (m - 1 / lam) * gamma + 1 / lam
    vp = (mp - 1 / pow(lam, 2)) * gamma + 1 / pow(lam, 2)

    phi = lam * m * v
    psi = (v - lam * vp) / gamma
    varphi = vp / pow(v, 2) - 1
    
    res = [m, mp, v, vp, phi, psi, varphi]
    return res

def th_stieltjes_func(Sigma, gamma, lam):
    xx = np.diag(Sigma)
    m = np.mean(1 / (xx + lam))
    mp = np.mean(1 / pow((xx + lam), 2))
    v = (m - 1 / lam) * gamma + 1 / lam
    vp = (mp - 1 / pow(lam, 2)) * gamma + 1 / pow(lam, 2)

    phi = lam * m * v
    psi = (v - lam * vp) / gamma
    varphi = vp / pow(v, 2) - 1
    
    res = [m, mp, v, vp, phi, psi, varphi]
    return res

def get_beta_func(X, Y, lam, eta, w):
    n = X.shape[0]
    p = X.shape[1]
    mu_hat = np.dot(np.transpose(X), (Y * 2 - np.ones(n)) / n)
    sigma_hat = ((sum(Y == 0) - 1) * np.cov(X[Y == 0,], rowvar = False) + (sum(Y == 1) - 1) * np.cov(X[Y == 1,], rowvar = False)) / (len(Y) - 2)
    beta_hat = LA.solve(sigma_hat + lam * np.diag(np.ones(p)), mu_hat + eta * w)
    return beta_hat

def pred_risk_func(X, Y, beta):
    Y_hat = (np.sign(np.dot(X, beta)) + 1) / 2
    err = np.mean(Y_hat != Y)
    return err
