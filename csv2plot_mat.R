source('./plot_functions.R')
library(ggplot2)

etas = seq(0, 50, 5) / 10
rhos = c(0.3,  0.6,  0.9)
alpha_ratios = c(0.5, 1, 2)
gammas = c(0.5, 2)
oracle_error = 0.01

get_mat_df_func = function(gamma, alpha_ratio, rho, etas) {
  # rho = 0.3
  # alpha_ratio = 0.5
  # gamma = 0.5
  
  theoret = NULL
  empir = NULL
  
  for(eta in etas) {
    filename = sprintf('./idcode/rv_data_simu/gamma_%.1f_ratio_%.1f_rho_%.1f_eta_%.1f_.csv', gamma, alpha_ratio, rho, eta)
    filename_th = sprintf('./idcode/rv_data_simu/gamma_%.1f_ratio_%.1f_rho_%.1f_eta_%.1f_th_.csv', gamma, alpha_ratio, rho, eta)
    empirics = read.csv(filename, header = T)
    theoretics = read.csv(filename_th, header = T)
    
    colnames(theoretics) = c('lambda','risk')
    theoretics$eta = rep(eta, nrow(theoretics))
    empirics$eta = rep(eta, nrow(empirics))
    
    theoret = rbind(theoret, theoretics)
    empir = rbind(empir, empirics)
  }

  rdf = rbind(theoret, empir)
  rdf$lambda = sqrt(rdf$lambda)
  rdf$eta = as.factor(rdf$eta)
  rdf$value = c(rep('Theoretical', nrow(theoret)), rep('Empirical', nrow(empir)))
  
  return(rdf)
}


for(gamma in gammas) {
  for(alpha_ratio in alpha_ratios) {
    for(rho in rhos) {
      print(paste('gamma =', gamma, 'alpha_ratio =', alpha_ratio, 'rho =', rho))
      rdf = get_mat_df_func(gamma, alpha_ratio, rho, etas)
      p = plot_result_func(rdf, rho, alpha_ratio, gamma, oracle_error)
      ggsave(
        filename = paste('./plots_nonoise_mat_png/plot_nonoise_rho', rho, 'ratio', alpha_ratio, 'gamma', gamma, '.png', sep = '_'),
        plot = p, width = 7, height = 4)
    }
  }
}