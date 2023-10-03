function [lambda,risk] = compute_tlda_risk(w,t,gamma,alpha,alpha_ratio,rho,eta)
%Compute the prediction error of Regularized Discriminant Analysis
%
%Inputs
%w,t - input spectral distribution is a mixture of point masses H = sum
%delta_{t_i} * w_i
%gamma - aspect ratio
%alpha - signal strength
%
%Output
%lambda - grid of regularization parameters
%risk - prediction error of RDA on the lambda grid

[lambda,m,v,~,v_prime] = compute_ST(w,t,gamma);

%% compute risk
alpha_s = alpha/alpha_ratio;
Theta = (alpha^2 + eta*rho*alpha*alpha_s)*m.*v.*lambda./sqrt((alpha^2 + eta^2*alpha_s^2 + 2*eta*rho*alpha_s*alpha)*(v-lambda.*v_prime)/gamma+(v_prime-v.^2)./v.^2);
risk = normcdf(-Theta);
end