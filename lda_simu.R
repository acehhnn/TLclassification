library(MASS)
library(ggplot2)
library(latex2exp)
source('lda_functions.R')

n_train = 50
n_test = 100
alpha_t = 1
root_lambdas = seq(1, 100) / 10
etas = seq(0, 50, 5) / 10
rhos = c(0.3, 0.8, 0.9)
alpha_ratios = c(2, 10/9)
gammas = c(2, 0.5)

seed_base = 20230912
seeds = seq(1, 500) + seed_base

lda_simu_func = function(seed, n_train, n_test, p, alpha_t, alpha_s, rho, eta) {
  # seed = 100
  # p = 100
  # alpha_t = 1
  # alpha_s = 2
  # rho = .3
  # eta = 0
  
  n = n_train + n_test
  set.seed(seed)
  Sigma = matrix(c(alpha_s^2, rho * alpha_s * alpha_t, rho * alpha_s * alpha_t, alpha_t^2) / p, nrow = 2)
  
  wt = mvrnorm(n = p, mu = rep(0, 2), Sigma = Sigma)
  w = wt[,1]
  tau = wt[,2]
  mu = mvrnorm(n = 1, mu = rep(0, p), Sigma = diag(alpha_t^2 / p, p))
  mu0 = mu - tau
  mu1 = mu + tau
  
  Y = sample(c(0, 1), size = n, replace = TRUE, prob = c(1/2, 1/2))
  X = sapply(Y, function(y) {
    mvrnorm(n = 1, mu = y * mu1 + (1 - y) * mu0, Sigma = diag(alpha_t^2, p))
  })
  X = t(X)
  
  # empirical
  # betas = lda_func(X, Y, root_lambdas^2, eta, w)$beta
  # empiricals = sapply(betas, function(beta) {
  #   pred_err_func(X, Y, beta)
  # })
  empiricals = sapply(root_lambdas, function(lambda){
    lda_res = lda_alt_func(X[1:n_train,], Y[1:n_train], lambda^2, eta, w)
    pred_err_func(X[(n_train + 1):n,], Y[(n_train + 1):n], lda_res$beta)
  })
  
  return(list(empirical = empiricals))
}

get_df_func = function(n_train, n_test, gamma, alpha_t, alpha_ratio, rho, etas) {
  p = floor(n_train * gamma)
  alpha_s = alpha_t / alpha_ratio
  
  theoret = NULL
  empir = NULL
  
  for(eta in etas) {
    theoretics = NULL
    empirics = NULL
    
    # theoretical
    ss = get_stieltjes_func(diag(alpha_t^2, p), n, p)
    # ss = get_risk_func(X)
    thetas = sapply(root_lambdas, function(lambda) {
      (alpha_t^2 + eta * rho * alpha_t * alpha_s) * ss$phi(lambda^2) / sqrt((alpha_t^2 + eta^2 * alpha_s^2 + 2 * eta * rho * alpha_t * alpha_s) * ss$psi(lambda^2) + ss$varphi(lambda^2))
    })
    theoretics = pnorm(-thetas)
    
    for(seed in seeds){
      ress = lda_simu_func(seed = seed, n_train = n_train, n_test = n_test, p = p, alpha_t = alpha_t, alpha_s = alpha_s, rho = rho, eta = eta)
      empirics = rbind(empirics, ress$empirical)
    }
    theoret = c(theoret, theoretics)
    empir = c(empir, colMeans(empirics))
  }
  
  df = data.frame(
    root_lambdas = rep(root_lambdas, length(etas)),
    the_risk = theoret,
    emp_risk = empir,
    etas = rep(etas, each = length(root_lambdas))
  )
  
  rdf = data.frame(
    root_lambdas = rep(df$root_lambdas, 2),
    risk = c(df$the_risk, df$emp_risk),
    etas = rep(as.factor(df$etas), 2),
    value = rep(c('Theoretical', 'Empirical'), each = nrow(df))
  )
  return(rdf)
}


plot_result_func = function(rdf, rho, alpha_ratio, gamma) {
  plot_obj = ggplot(rdf, aes(x = root_lambdas, y = risk, color = etas, linetype = value)) + 
    geom_line() + theme_minimal() + scale_color_brewer(palette = 'RdYlBu') +
    labs(y = 'Prediction Error', title = paste('rho =', rho, 'ratio =', alpha_ratio, 'gamma =', gamma), x = expression(sqrt(lambda)), color = expression(eta))
  plot_obj
  return(plot_obj)
}


plots = list()
for(alpha_ratio in alpha_ratios) {
  for(gamma in gammas) {
    for(rho in rhos) {
      print(paste('alpha_ratio =', alpha_ratio, 'gamma =', gamma, 'rho =', rho))
      rdf = get_df_func(n_train, n_test, gamma, oracle_error_rate, alpha_ratio, rho, etas)
      plots = append(plots, plot_result_func(rdf, rho, alpha_ratio, gamma))
    }
  }
}

ggsave(
  filename = "plots_1.pdf", 
  plot = marrangeGrob(plots, nrow = 1, ncol = 1), 
  width = 15, height = 9
)
