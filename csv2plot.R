library(ggplot2)

rhos = c(0.3,  0.6,  0.9)
alpha_ratios = c(0.5, 1, 2)
gammas = c(0.5, 2)

plot_result_func = function(rdf, rho, alpha_ratio, gamma) {
  plot_obj = ggplot(rdf, aes(x = root_lambdas, y = risk, color = etas, linetype = value), alpha = value) + 
    geom_line() + theme_minimal() + scale_color_brewer(palette = 'RdYlBu') + 
    geom_hline(yintercept = pnorm(-1), linetype = 3) +
    labs(y = 'Prediction Error', title = paste('rho =', rho, 'ratio =', alpha_ratio, 'gamma =', gamma), 
         x = expression(sqrt(lambda)), color = expression(eta), linetype = NULL, alpha = NULL) + 
    scale_alpha_manual(c(0.7, 1))
  plot_obj
  return(plot_obj)
}

for(alpha_ratio in alpha_ratios) {
  for(gamma in gammas) {
    for(rho in rhos) {
      print(paste('alpha_ratio =', alpha_ratio, 'gamma =', gamma, 'rho =', rho))
      rdf = read.csv(paste('./data_nonoise_emp/rho', rho, 'ratio', alpha_ratio, 'gamma', gamma, '.csv', sep = '_'))
      rdf$etas = as.factor(rdf$etas)
      p = plot_result_func(rdf, rho, alpha_ratio, gamma)
      ggsave(
        filename = paste('./plot_nonoise_emp/rho', rho, 'ratio', alpha_ratio, 'gamma', gamma, '.pdf', sep = '_'),
        plot = p, width = 7, height = 4)
    }
  }
}