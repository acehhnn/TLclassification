import numpy as np
import pandas as pd
import random
import math
import os
from scipy.stats import norm
import tlclass_functions as tlfun

n_train = 50
n_test = 100
alpha_t = 1
root_lambdas = np.linspace(0.01, 1, 100)
etas = np.linspace(0, 5, 11)
rhos = [0.3,  0.6,  0.9]
alpha_ratios = [0.5, 1, 2]
gammas = [0.5, 2]

seed_base = 20230912
seeds = np.linspace(1, 500, 500) + seed_base

def lda_simu_func(seed, n_train, n_test, p, alpha_t, alpha_s, rho, eta):
    # seed = 100
    # p = 100
    # alpha_t = 1
    # alpha_s = 2
    # rho = .3
    # eta = 0
    
    n = n_train + n_test
    random.seed(seed)
    Sigma = [[alpha_s ** 2 / p, rho * alpha_s * alpha_t / p], [rho * alpha_s * alpha_t / p, alpha_t ** 2 / p]]
    
    wt = np.random.multivariate_normal(np.zeros(2), Sigma, p)
    w = wt[:, 0]
    tau = wt[:, 1]

    X = []
    Y = np.random.binomial(1, 0.5, n)
    for i in range(len(Y)):
        y = Y[i]
        if i == 0:
            X = np.random.multivariate_normal(np.multiply((y * 2 - 1), tau), np.diag(np.full(p, alpha_t ** 2)), 1)
        else:
            X = np.row_stack((X, np.random.multivariate_normal(np.multiply((y * 2 - 1), tau), np.diag(np.full(p, alpha_t ** 2)), 1)))
    
    # theoretical
    theoreticals = []
    for  lam in root_lambdas:
        ss = tlfun.emp_stieltjes_func(X[range(n_train), :], Y[range(n_train)], pow(lam,2))
        theta = (alpha_t ** 2 + eta * rho * alpha_t * alpha_s) * ss[4] / math.sqrt((alpha_t ** 2 + eta ** 2 * alpha_s ** 2 + 2 * eta * rho * alpha_t * alpha_s) * ss[5] + ss[6])
        theoretical = norm.cdf(-theta)
        theoreticals = np.append(theoreticals, theoretical)
    
    # empirical
    empiricals = []
    for lam in root_lambdas:
        beta = tlfun.get_beta_func(X[range(n_train), :], Y[range(n_train)], pow(lam,2), eta, w)
        empirical = tlfun.pred_risk_func(X[range(n_train, n), :], Y[range(n_train,n)], beta)
        empiricals = np.append(empiricals, empirical)
    
    res = pd.DataFrame({'empiricals': empiricals, 'theoreticals': theoreticals})
    # print('res.empiricals =', np.array(res.loc[:, 'empiricals']))
    return res


def get_df_func(n_train, n_test, gamma, alpha_t, alpha_ratio, rho, etas):
    # n_train = 50
    # n_test = 100
    # gamma = 0.5
    # alpha_t = 1
    # alpha_ratio = 0.5
    # rho = 0.3

    p = math.floor(n_train * gamma)
    alpha_s = alpha_t / alpha_ratio
    
    theoret = []
    empir = []
    
    for eta in etas:
        # # theoretical
        # if (gamma <= 1):
        #     x_Sigma = np.diag(alpha_t**2, p)
        # else:
        #     x_Sigma = np.diag([np.full(n_train, alpha_t**2), np.zeros(p - n_train)].flatten())
        
        # theoretics = []
        # for lam in root_lambdas:
        #     ss = tlfun.th_stieltjes_func(x_Sigma, gamma, lam)
        #     theta = (alpha_t**2 + eta * rho * alpha_t * alpha_s) * ss[4] / math.sqrt((alpha_t**2 + eta**2 * alpha_s**2 + 2 * eta * rho * alpha_t * alpha_s) * ss[5] + ss[6])
        #     theoretics = norm.pdf(-theta)
        
        for i in range(len(seeds)):
            seed = seeds[i]
            ress = lda_simu_func(seed = seed, n_train = n_train, n_test = n_test, p = p, alpha_t = alpha_t, alpha_s = alpha_s, rho = rho, eta = eta)
            if i == 0:
                theoretics = np.expand_dims(ress.loc[:, 'theoreticals'], axis = 0)
                empirics = np.expand_dims(ress.loc[:, 'empiricals'], axis = 0)
            else:
                theoretics = np.append(theoretics, np.expand_dims(ress.loc[:, 'theoreticals'], axis = 0), axis = 0)
                empirics = np.append(empirics, np.expand_dims(ress.loc[:, 'empiricals'], axis = 0), axis = 0)
        # print('theoretics.ncol =', theoretics.shape[1], 'theoretics.nrow =', theoretics.shape[0])

        theoret = np.append(theoret, np.average(theoretics, axis = 0))
        empir = np.append(empir, np.average(empirics, axis = 0))

    # print('theoret:', len(theoret), 'empir', len(empir))
    # print(np.tile(root_lambdas, (1, len(etas))).flatten())
    
    df = pd.DataFrame({
        'root_lambdas': (np.tile(root_lambdas, (1, len(etas))).flatten()),
        'the_risk': theoret,
        'emp_risk': empir,
        'etas': np.repeat(etas, len(root_lambdas))
    })
    
    rdf = pd.DataFrame({
        'root_lambdas': (np.tile(df.loc[:, 'root_lambdas'], (1, 2))).flatten(),
        'risk': np.array([df.loc[:, 'the_risk'], df.loc[:, 'emp_risk']]).flatten(),
        'etas': (np.tile(df.loc[:, 'etas'], (1, 2))).flatten(),
        'value': np.repeat(['Theoretical', 'Empirical'], df.shape[0])
    })
    return rdf
path = 'D:/THU/stuabd/resch/dr/LearningTheory/code/py/data_nonoise_emp'
if not os.path.exists(path):
    os.makedirs(path)

for alpha_ratio in alpha_ratios:
    for gamma in gammas:
        for rho in rhos:
            print('alpha_ratio =', alpha_ratio, 'gamma =', gamma, 'rho =', rho)
            rdf = get_df_func(n_train, n_test, gamma, alpha_t, alpha_ratio, rho, etas)
            rdf.to_csv(os.path.join(path, 'rho_' + str(rho) + '_ratio_' + str(alpha_ratio) + '_gamma_' + str(gamma) + '_.csv'), index = False, header = True)
