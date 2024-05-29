function [loss,A_GF9,phi_GF9,p_start_GF9]= loss_cross_entropy_gmm9mus(w, X, y_true,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0)

epsilon = 1e-15;  

X_NS0_WF{1} = X_NS0*NS0_w0;
X_DF5_WF{1} = X_DF5*DF5_w0;
X_FF6_WF{1} = X_FF6*FF6_w0;
X_AF7_WF{1} = X_AF7*AF7_w0;
X_SPF8_WF{1} = X_SPF8*SPF8_w0;
X_GF9_WF{1} = X_GF9*w;
% X_GF9_WF{1} = X_GF9*GF9_w0;

Q = 2;      % state num
M = 3;      % mix num 
% Q = 1;      % state num
% M = 1;      % mix num 
[p_start_NS0, A_NS0, phi_NS0, ~] = ChmmGmm(X_NS0_WF, Q, M);
[p_start_DF5, A_DF5, phi_DF5, ~] = ChmmGmm(X_DF5_WF, Q, M);
[p_start_FF6, A_FF6, phi_FF6, ~] = ChmmGmm(X_FF6_WF, Q, M);
[p_start_AF7, A_AF7, phi_AF7, ~] = ChmmGmm(X_AF7_WF, Q, M);
[p_start_SPF8, A_SPF8, phi_SPF8, ~] = ChmmGmm(X_SPF8_WF, Q, M);
[p_start_GF9, A_GF9, phi_GF9, ~] = ChmmGmm(X_GF9_WF, Q, M);

X_WF_validate_NS0 = X*NS0_w0;
X_WF_validate_DF5 = X*DF5_w0;
X_WF_validate_FF6 = X*FF6_w0;
X_WF_validate_AF7 = X*AF7_w0;
X_WF_validate_SPF8 = X*SPF8_w0;
X_WF_validate_GF9 = X*w;

test_number = size(X_WF_validate_NS0,1);
test_number_range = test_number-10+1;
X_WF_test_new_NS0 = zeros(test_number_range,10);
X_WF_test_new_DF5 = zeros(test_number_range,10);
X_WF_test_new_FF6 = zeros(test_number_range,10);
X_WF_test_new_AF7 = zeros(test_number_range,10);
X_WF_test_new_SPF8 = zeros(test_number_range,10);
X_WF_test_new_GF9 = zeros(test_number_range,10);
for a = 1:test_number_range
    X_WF_test_new_NS0(a,1:end) = X_WF_validate_NS0(a:a+9);
    X_WF_test_new_DF5(a,1:end) = X_WF_validate_DF5(a:a+9);
    X_WF_test_new_FF6(a,1:end) = X_WF_validate_FF6(a:a+9);
    X_WF_test_new_AF7(a,1:end) = X_WF_validate_AF7(a:a+9);
    X_WF_test_new_SPF8(a,1:end) = X_WF_validate_SPF8(a:a+9);
    X_WF_test_new_GF9(a,1:end) = X_WF_validate_GF9(a:a+9);
end
N_test_NS0 = size(X_WF_test_new_NS0,1);   
alpha_NS0 = zeros(N_test_NS0, 1); 
alpha_DF5 = zeros(N_test_NS0, 1); 
alpha_FF6 = zeros(N_test_NS0, 1);
alpha_AF7 = zeros(N_test_NS0, 1);
alpha_SPF8 = zeros(N_test_NS0, 1); 
alpha_GF9 = zeros(N_test_NS0, 1); 
for b = 1:test_number_range
    logp_xn_given_zn_NS0 = Gmm_logp_xn_given_zn(X_WF_test_new_NS0(b,1:end), phi_NS0);
    [~,~, loglik_NS0] = LogForwardBackward(logp_xn_given_zn_NS0, p_start_NS0, A_NS0);
    alpha_NS0(b, 1) = loglik_NS0;
    
    logp_xn_given_zn_DF5 = Gmm_logp_xn_given_zn(X_WF_test_new_DF5(b,1:end), phi_DF5);
    [~,~, loglik_DF5] = LogForwardBackward(logp_xn_given_zn_DF5, p_start_DF5, A_DF5);
    alpha_DF5(b, 1) = loglik_DF5;
    logp_xn_given_zn_FF6 = Gmm_logp_xn_given_zn(X_WF_test_new_FF6(b,1:end), phi_FF6);
    [~,~, loglik_FF6] = LogForwardBackward(logp_xn_given_zn_FF6, p_start_FF6, A_FF6);
    alpha_FF6(b, 1) = loglik_FF6;
    logp_xn_given_zn_AF7 = Gmm_logp_xn_given_zn(X_WF_test_new_AF7(b,1:end), phi_AF7);
    [~,~, loglik_AF7] = LogForwardBackward(logp_xn_given_zn_AF7, p_start_AF7, A_AF7);
    alpha_AF7(b, 1) = loglik_AF7;
    logp_xn_given_zn_SPF8 = Gmm_logp_xn_given_zn(X_WF_test_new_SPF8(b,1:end), phi_SPF8);
    [~,~, loglik_SPF8] = LogForwardBackward(logp_xn_given_zn_SPF8, p_start_SPF8, A_SPF8);
    alpha_SPF8(b, 1) = loglik_SPF8;
    logp_xn_given_zn_GF9 = Gmm_logp_xn_given_zn(X_WF_test_new_GF9(b,1:end), phi_GF9);
    [~,~, loglik_GF9] = LogForwardBackward(logp_xn_given_zn_GF9, p_start_GF9, A_GF9);
    alpha_GF9(b, 1) = loglik_GF9;
end

probs = [alpha_NS0 alpha_DF5 alpha_FF6 alpha_AF7 alpha_SPF8 alpha_GF9];
y_pred = min_max_normalize_columns(probs', epsilon, 1)';

y_pred = y_pred ./ sum(y_pred, 2);             

loss = -(1/N_test_NS0) * sum(sum(y_true .* log(y_pred), 2));

end
