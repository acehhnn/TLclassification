%% this code is a wrapper to perform simulations with an identical model for transfer learning
% the signal strength is normalized so that the oracle achieves 1% error
addpath("D:/THU/stuabd/resch/dr/LearningTheory/code/high-dim-risk-experiments-master/high-dim-risk-experiments-master/Code")
addpath("D:/THU/stuabd/resch/dr/LearningTheory/code/idcode")

%%  normalize the signal strength to have the same oracle mis-classification error
%run AR(1) simulations for a grid of parameters

%define grids
%exp_lambda = [1/10; 1/4; 1; 4];
%gamma = [0.5; 1; 2];
%oracle_error_rate =0.01; %[0.01 0.02 0.05];
%% test case
gamma = [0.5; 2];
alpha = 1;
oracle_error_rate = 0.01;
eta = [0; 0.5; 1; 1.5; 2; 2.5; 3; 3.5; 4; 4.5; 5];
rho = [0.3; 0.6; 0.9];
alpha_ratio = [0.5; 1; 2];

%% loop over parameters
a = {'-','--',':','-.'};
n_lambda = 100;
n = 50;
for i=1:length(alpha)
    for j=1:length(gamma)
        for k=1:length(oracle_error_rate)
            for l=1:length(alpha_ratio)
                for m=1:length(rho)
                    for t=1:length(eta)
                        %perform experiment
                        
                        [lambda,risk,lambda_th,risk_th] = run_id_sim_norm_sig(alpha(i),gamma(j),oracle_error_rate(k),alpha_ratio(l),rho(m),eta(t),n_lambda,n);
                        %%
                        
                        %save results
                        columns = {'lambda','risk'};
                        columns_th = {'lambda_th','risk_th'};
                        data = table(lambda,risk,'VariableNames',columns);
                        data_th = table(lambda_th,risk_th,'VariableNames',columns_th);
                        filename = sprintf('./rv_data_simu/gamma_%.1f_ratio_%.1f_rho_%.1f_eta_%.1f_.csv',gamma(j),alpha_ratio(l),rho(m),eta(t));
                        filename_th = sprintf('./rv_data_simu/gamma_%.1f_ratio_%.1f_rho_%.1f_eta_%.1f_th_.csv',gamma(j),alpha_ratio(l),rho(m),eta(t));
                        writetable(data,filename);
                        fprintf(['Saved Results to ' filename '\n']);
                        writetable(data_th,filename_th);
                        fprintf(['Saved Results to' filename_th '\n']);
                    end
                end
            end
        end
    end
end
