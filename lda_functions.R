get_stieltjes_func = function(Sigma, n, p) {
  xx = eigen(Sigma, symmetric = TRUE, only.values = TRUE)$values
  # print(xx)
  gamma = p / n
  
  m = Vectorize(function(lambda) {
    mean(1 / (xx + lambda))
  })
  mp = Vectorize(function(lambda) {
    mean(1 / (xx + lambda)^2)
  })
  v = Vectorize(function(lambda) {
    (m(lambda) - 1/lambda) * gamma + 1/lambda
  })
  vp = Vectorize(function(lambda) {
    (mp(lambda) - 1/lambda^2) * gamma + 1/lambda^2
  })
  
  phi = Vectorize(function(lambda) {
    lambda * m(lambda) * v(lambda)
  })
  psi = Vectorize(function(lambda) {
    (v(lambda) - lambda * vp(lambda)) / gamma
  })
  varphi = Vectorize(function(lambda) {
    vp(lambda) / v(lambda)^2 - 1
  })
  
  return(list(m = m, v = v, vp = vp, phi = phi, psi = psi, varphi = varphi))
}

get_risk_func = function(X) {
  
  M = X %*% t(X)
  xx = svd(M, nu = 0, nv = 0)$d
  gamma = ncol(X) / nrow(X)
  
  v = Vectorize(function(lambda) {
    mean(1 / (xx + lambda))
  })
  vp = Vectorize(function(lambda) {
    mean(1 / (xx + lambda)^2)
  })
  m = Vectorize(function(lambda) {
    (v(lambda) - 1/lambda) / gamma + 1/lambda
  })
  
  phi = Vectorize(function(lambda) {
    lambda * m(lambda) * v(lambda)
  })
  psi = Vectorize(function(lambda) {
    (v(lambda) - lambda * vp(lambda)) / gamma
  })
  varphi = Vectorize(function(lambda) {
    vp(lambda) / v(lambda)^2 - 1
  })
  
  return(list(m = m, v = v, vp = vp, phi = phi, psi = psi, varphi = varphi))
}

lda_func = function(X, Y, lambda, eta, w) {
  mu0 = colMeans(X[Y==0,])
  mu1 = colMeans(X[Y==1,])
  mu = (mu0 + mu1) / 2
  tau = (mu1 - mu0) / 2
  
  sigma = ((sum(Y==0) - 1) * var(X[Y==0,]) + (sum(Y==1) - 1) * var(X[Y==0,])) / (length(Y) - 2)	
  sigma.svd = svd(sigma)
  
  beta = sigma.svd$v %*% diag(1 / (sigma.svd$d + lambda)) %*% t(sigma.svd$u) %*% (tau + eta * w)
  
  return(list(mu = mu, tau = tau, beta = beta))
}

lda_alt_func = function(X, Y, lambda, eta, w){
  n = nrow(X)
  p = ncol(X)
  X_sgn = X * ( matrix((2 * Y - 1), n, 1) %*% matrix(1, 1, p))
  mu_hat = colMeans(X_sgn)
  sigma_hat = var(X_sgn)
  beta_hat = solve(sigma_hat + lambda * diag(rep(1, p)), mu_hat + eta * w)
  
  return(list(beta = beta_hat))
}

pred_err_func = function(X, Y, beta) {
  # muM = matrix(rep(mu, length(Y)), byrow = TRUE, nrow = length(Y))
  # print(muM)
  # Y_hat = (sign((X - muM) %*% beta) + 1) / 2
  Y_hat = (sign(X %*% beta) + 1) / 2
  
  err = mean(Y_hat != Y)
  
  return(err)
}