clear
clc
close all
tic
% ===========================================================
data_NS0=importdata('D:\cw\code0320\data0303\0 normal state\pos_x_perdeal.txt');
data_DF5=importdata('D:\cw\code0320\data0303\5 DF\pos_x_perdeal.txt');
data_FF6=importdata('D:\cw\code0320\data0303\6 FF\pos_x_perdeal.txt');
data_AF7=importdata('D:\cw\code0320\data0303\7 AF\pos_x_perdeal.txt');
data_SPF8=importdata('D:\cw\code0320\data0303\8 SPF\pos_x_perdeal.txt');
data_GF9=importdata('D:\cw\code0320\data0303\9 GF\pos_x_perdeal.txt');
data_combine = [data_NS0;data_DF5;data_FF6;data_AF7;data_SPF8;data_GF9];
data_combine_normalized = min_max_normalize_columns(data_combine(1:end,2:13), 0, 1);
each_data_number = 3961;
x_normalized_NS0 = data_combine_normalized(1:each_data_number,1:end);
x_normalized_DF5 = data_combine_normalized(each_data_number+1:each_data_number*2,1:end);
x_normalized_FF6 = data_combine_normalized(each_data_number*2+1:each_data_number*3,1:end);
x_normalized_AF7 = data_combine_normalized(each_data_number*3+1:each_data_number*4,1:end);
x_normalized_SPF8 = data_combine_normalized(each_data_number*4+1:each_data_number*5,1:end);
x_normalized_GF9 = data_combine_normalized(each_data_number*5+1:each_data_number*6,1:end);

training_number = 900;
test_number = 225;

X_NS0 = x_normalized_NS0(1:training_number,1:end);
X_NS0_test = x_normalized_NS0(training_number+1:end,1:end);
X_NS0_test = X_NS0_test(1:test_number,1:end);

X_NS0_lab = data_NS0(1:training_number-9,14);

X_DF5 = x_normalized_DF5(1:training_number,1:end);

X_DF5_test = x_normalized_DF5(training_number+1:end,1:end);
X_DF5_test = X_DF5_test(1:test_number,1:end);
X_DF5_lab = data_DF5(1:training_number-9,14);

X_FF6 = x_normalized_FF6(1:training_number,1:end);

X_FF6_test = x_normalized_FF6(training_number+1:end,1:end);
X_FF6_test = X_FF6_test(1:test_number,1:end);
X_FF6_lab = data_FF6(1:training_number-9,14);

X_AF7 = x_normalized_AF7(1:training_number,1:end);

X_AF7_test = x_normalized_AF7(training_number+1:end,1:end);
X_AF7_test = X_AF7_test(1:test_number,1:end);
X_AF7_lab = data_AF7(1:training_number-9,14);

X_SPF8 = x_normalized_SPF8(1:training_number,1:end);

X_SPF8_test = x_normalized_SPF8(training_number+1:end,1:end);
X_SPF8_test = X_SPF8_test(1:test_number,1:end);
X_SPF8_lab = data_SPF8(1:training_number-9,14);

X_GF9 = x_normalized_GF9(1:training_number,1:end);

X_GF9_test = x_normalized_GF9(training_number+1:end,1:end);
X_GF9_test = X_GF9_test(1:test_number,1:end);
X_GF9_lab = data_GF9(1:training_number-9,14);

zeros_fl = zeros(891,1);

X_NS0_lab_zs = [X_NS0_lab+1 zeros_fl zeros_fl zeros_fl zeros_fl zeros_fl]/1;
X_DF5_lab_zs = [zeros_fl X_DF5_lab zeros_fl zeros_fl zeros_fl zeros_fl]/5;
X_FF6_lab_zs = [zeros_fl zeros_fl X_FF6_lab zeros_fl zeros_fl zeros_fl]/6;
X_AF7_lab_zs = [zeros_fl zeros_fl zeros_fl X_AF7_lab zeros_fl zeros_fl]/7;
X_SPF8_lab_zs = [zeros_fl zeros_fl zeros_fl zeros_fl X_SPF8_lab zeros_fl]/8;
X_GF9_lab_zs = [zeros_fl zeros_fl zeros_fl zeros_fl zeros_fl X_GF9_lab]/9;

NS0_w0=[1,1,1,1,1,1,1,1,1,1,1,1]'/12;
DF5_w0=[1,1,1,1,1,1,1,1,1,1,1,1]'/12;
FF6_w0=[1,1,1,1,1,1,1,1,1,1,1,1]'/12;
AF7_w0=[1,1,1,1,1,1,1,1,1,1,1,1]'/12;
SPF8_w0=[1,1,1,1,1,1,1,1,1,1,1,1]'/12;
GF9_w0=[1,1,1,1,1,1,1,1,1,1,1,1]'/12;

options = optimoptions('fmincon','MaxIter',1,'algorithm','interior-point','StepTolerance', 0.001);

A=[0,0,0,0,0,0,0,0,0,0,0,0];
b=[0];
Aeq=[1,1,1,1,1,1,1,1,1,1,1,1];
beq=[1];
LB = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
UB = [1,1,1,1,1,1,1,1,1,1,1,1];
max_diedai = 100;
acc_for = zeros(max_diedai,1);
loss_all_vector = zeros(max_diedai,1);
epi = 1e-15;
for cishu =1:max_diedai
    fun=@(w)loss_cross_entropy_gmm0mus(w, X_NS0, X_NS0_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
    [NS0_w1, NS0_fval1, NS0_exitflag1, NS0_output1] = fmincon(fun,NS0_w0,A,b,Aeq,beq,LB,UB,[],options);
    [loss_NS0,A_NS0,phi_NS0,p_start_NS0] = loss_cross_entropy_gmm0mus(NS0_w1, X_NS0, X_NS0_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
 
    fun=@(w)loss_cross_entropy_gmm5mus(w, X_DF5, X_DF5_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
    [DF5_w1, DF5_fval1, DF5_exitflag1, DF5_output1] = fmincon(fun,DF5_w0,A,b,Aeq,beq,LB,UB,[],options);
    [loss_DF5,A_DF5,phi_DF5,p_start_DF5] = loss_cross_entropy_gmm5mus(DF5_w1, X_DF5, X_DF5_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);

    fun=@(w)loss_cross_entropy_gmm6mus(w, X_FF6, X_FF6_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
    [FF6_w1, FF6_fval1, FF6_exitflag1, FF6_output1] = fmincon(fun,FF6_w0,A,b,Aeq,beq,LB,UB,[],options);
    [loss_FF6,A_FF6,phi_FF6,p_start_FF6] = loss_cross_entropy_gmm6mus(FF6_w1, X_FF6, X_FF6_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);

    fun=@(w)loss_cross_entropy_gmm7mus(w, X_AF7, X_AF7_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
    [AF7_w1, AF7_fval1, AF7_exitflag1, AF7_output1] = fmincon(fun,AF7_w0,A,b,Aeq,beq,LB,UB,[],options);
    [loss_AF7,A_AF7,phi_AF7,p_start_AF7] = loss_cross_entropy_gmm7mus(AF7_w1, X_AF7, X_AF7_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);

    fun=@(w)loss_cross_entropy_gmm8mus(w, X_SPF8, X_SPF8_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
    [SPF8_w1,SPF8_fval1, SPF8_exitflag1, SPF8_output1] = fmincon(fun,SPF8_w0,A,b,Aeq,beq,LB,UB,[],options);
    [loss_SPF8,A_SPF8,phi_SPF8,p_start_SPF8] = loss_cross_entropy_gmm8mus(SPF8_w1, X_SPF8, X_SPF8_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);

    fun=@(w)loss_cross_entropy_gmm9mus(w, X_GF9, X_GF9_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
    [GF9_w1, GF9_fval1, GF9_exitflag1, GF9_output1] = fmincon(fun,GF9_w0,A,b,Aeq,beq,LB,UB,[],options);
    [loss_GF9,A_GF9,phi_GF9,p_start_GF9] = loss_cross_entropy_gmm9mus(GF9_w1, X_GF9, X_GF9_lab_zs,X_NS0,NS0_w0,X_DF5,DF5_w0,X_FF6,FF6_w0,X_AF7,AF7_w0,X_SPF8,SPF8_w0,X_GF9,GF9_w0);
    loss_all = loss_NS0 + loss_DF5 + loss_FF6 + loss_AF7 + loss_SPF8 + loss_GF9;
    loss_all_vector(cishu) = loss_all;
    epsilon = 1e-15;
    count_raw = zeros(6,6);
    for  k = 1:6
    if k == 1
        
        X_NS0_WF_validate = X_NS0_test*NS0_w1;
        X_DF5_WF_validate = X_NS0_test*DF5_w1;
        X_FF6_WF_validate = X_NS0_test*FF6_w1;
        X_AF7_WF_validate = X_NS0_test*AF7_w1;
        X_SPF8_WF_validate = X_NS0_test*SPF8_w1;
        X_GF9_WF_validate = X_NS0_test*GF9_w1;

test_number = size(X_NS0_WF_validate,1);
test_number_range = test_number-10+1;
X_NS0_WF_validate_new = zeros(test_number_range,10);
X_DF5_WF_validate_new = zeros(test_number_range,10);
X_FF6_WF_validate_new = zeros(test_number_range,10);
X_AF7_WF_validate_new = zeros(test_number_range,10);
X_SPF8_WF_validate_new = zeros(test_number_range,10);
X_GF9_WF_validate_new = zeros(test_number_range,10);
for a = 1:test_number_range
    X_NS0_WF_validate_new(a,1:end) = X_NS0_WF_validate(a:a+9);
    X_DF5_WF_validate_new(a,1:end) = X_DF5_WF_validate(a:a+9);
    X_FF6_WF_validate_new(a,1:end) = X_FF6_WF_validate(a:a+9);
    X_AF7_WF_validate_new(a,1:end) = X_AF7_WF_validate(a:a+9);
    X_SPF8_WF_validate_new(a,1:end) = X_SPF8_WF_validate(a:a+9);
    X_GF9_WF_validate_new(a,1:end) = X_GF9_WF_validate(a:a+9); 
end
N_test_NS0 = size(X_NS0_WF_validate_new,1);
alpha_NS0 = zeros(N_test_NS0, 1);
alpha_DF5 = zeros(N_test_NS0, 1);
alpha_FF6 = zeros(N_test_NS0, 1);
alpha_AF7 = zeros(N_test_NS0, 1);
alpha_SPF8 = zeros(N_test_NS0, 1);
alpha_GF9 = zeros(N_test_NS0, 1);
for b = 1:test_number_range

    logp_xn_given_zn_NS0 = Gmm_logp_xn_given_zn(X_NS0_WF_validate_new(b,1:end), phi_NS0);
    [~,~, loglik_NS0] = LogForwardBackward(logp_xn_given_zn_NS0, p_start_NS0, A_NS0);
    alpha_NS0(b, 1) = loglik_NS0;
    logp_xn_given_zn_DF5 = Gmm_logp_xn_given_zn(X_DF5_WF_validate_new(b,1:end), phi_DF5);
    [~,~, loglik_DF5] = LogForwardBackward(logp_xn_given_zn_DF5, p_start_DF5, A_DF5);
    alpha_DF5(b, 1) = loglik_DF5;
    logp_xn_given_zn_FF6 = Gmm_logp_xn_given_zn(X_FF6_WF_validate_new(b,1:end), phi_FF6);
    [~,~, loglik_FF6] = LogForwardBackward(logp_xn_given_zn_FF6, p_start_FF6, A_FF6);
    alpha_FF6(b, 1) = loglik_FF6;
    logp_xn_given_zn_AF7 = Gmm_logp_xn_given_zn(X_AF7_WF_validate_new(b,1:end), phi_AF7);
    [~,~, loglik_AF7] = LogForwardBackward(logp_xn_given_zn_AF7, p_start_AF7, A_AF7);
    alpha_AF7(b, 1) = loglik_AF7;
    logp_xn_given_zn_SPF8 = Gmm_logp_xn_given_zn(X_SPF8_WF_validate_new(b,1:end), phi_SPF8);
    [~,~, loglik_SPF8] = LogForwardBackward(logp_xn_given_zn_SPF8, p_start_SPF8, A_SPF8);
    alpha_SPF8(b, 1) = loglik_SPF8;
    logp_xn_given_zn_GF9 = Gmm_logp_xn_given_zn(X_GF9_WF_validate_new(b,1:end), phi_GF9);
    [~,~, loglik_GF9] = LogForwardBackward(logp_xn_given_zn_GF9, p_start_GF9, A_GF9);
    alpha_GF9(b, 1) = loglik_GF9;
end

probs = [alpha_NS0 alpha_DF5 alpha_FF6 alpha_AF7 alpha_SPF8 alpha_GF9];
y_pred = min_max_normalize_columns(probs', epsilon, 1)';

for i =1:N_test_NS0
    for j = 1:6
        if y_pred(i,j)==1
            count_raw(1,j) = count_raw(1,j)+1;
        end
    end
end
    end
    if k ==2
        
X_NS0_WF_validate = X_DF5_test*NS0_w1;
X_DF5_WF_validate = X_DF5_test*DF5_w1;
X_FF6_WF_validate = X_DF5_test*FF6_w1;
X_AF7_WF_validate = X_DF5_test*AF7_w1;
X_SPF8_WF_validate = X_DF5_test*SPF8_w1;
X_GF9_WF_validate = X_DF5_test*GF9_w1;
      
test_number = size(X_NS0_WF_validate,1);
test_number_range = test_number-10+1;
X_NS0_WF_validate_new = zeros(test_number_range,10);
X_DF5_WF_validate_new = zeros(test_number_range,10);
X_FF6_WF_validate_new = zeros(test_number_range,10);
X_AF7_WF_validate_new = zeros(test_number_range,10);
X_SPF8_WF_validate_new = zeros(test_number_range,10);
X_GF9_WF_validate_new = zeros(test_number_range,10);
for a = 1:test_number_range
    X_NS0_WF_validate_new(a,1:end) = X_NS0_WF_validate(a:a+9);
    X_DF5_WF_validate_new(a,1:end) = X_DF5_WF_validate(a:a+9);
    X_FF6_WF_validate_new(a,1:end) = X_FF6_WF_validate(a:a+9);
    X_AF7_WF_validate_new(a,1:end) = X_AF7_WF_validate(a:a+9);
    X_SPF8_WF_validate_new(a,1:end) = X_SPF8_WF_validate(a:a+9);
    X_GF9_WF_validate_new(a,1:end) = X_GF9_WF_validate(a:a+9); 
end
N_test_NS0 = size(X_NS0_WF_validate_new,1);
alpha_NS0 = zeros(N_test_NS0, 1); 
alpha_DF5 = zeros(N_test_NS0, 1); 
alpha_FF6 = zeros(N_test_NS0, 1); 
alpha_AF7 = zeros(N_test_NS0, 1); 
alpha_SPF8 = zeros(N_test_NS0, 1); 
alpha_GF9 = zeros(N_test_NS0, 1); 
for b = 1:test_number_range
    logp_xn_given_zn_NS0 = Gmm_logp_xn_given_zn(X_NS0_WF_validate_new(b,1:end), phi_NS0);
    [~,~, loglik_NS0] = LogForwardBackward(logp_xn_given_zn_NS0, p_start_NS0, A_NS0);
    alpha_NS0(b, 1) = loglik_NS0;
    logp_xn_given_zn_DF5 = Gmm_logp_xn_given_zn(X_DF5_WF_validate_new(b,1:end), phi_DF5);
    [~,~, loglik_DF5] = LogForwardBackward(logp_xn_given_zn_DF5, p_start_DF5, A_DF5);
    alpha_DF5(b, 1) = loglik_DF5;
    logp_xn_given_zn_FF6 = Gmm_logp_xn_given_zn(X_FF6_WF_validate_new(b,1:end), phi_FF6);
    [~,~, loglik_FF6] = LogForwardBackward(logp_xn_given_zn_FF6, p_start_FF6, A_FF6);
    alpha_FF6(b, 1) = loglik_FF6;
    logp_xn_given_zn_AF7 = Gmm_logp_xn_given_zn(X_AF7_WF_validate_new(b,1:end), phi_AF7);
    [~,~, loglik_AF7] = LogForwardBackward(logp_xn_given_zn_AF7, p_start_AF7, A_AF7);
    alpha_AF7(b, 1) = loglik_AF7;
    logp_xn_given_zn_SPF8 = Gmm_logp_xn_given_zn(X_SPF8_WF_validate_new(b,1:end), phi_SPF8);
    [~,~, loglik_SPF8] = LogForwardBackward(logp_xn_given_zn_SPF8, p_start_SPF8, A_SPF8);
    alpha_SPF8(b, 1) = loglik_SPF8;
    logp_xn_given_zn_GF9 = Gmm_logp_xn_given_zn(X_GF9_WF_validate_new(b,1:end), phi_GF9);
    [~,~, loglik_GF9] = LogForwardBackward(logp_xn_given_zn_GF9, p_start_GF9, A_GF9);
    alpha_GF9(b, 1) = loglik_GF9;
end

probs = [alpha_NS0 alpha_DF5 alpha_FF6 alpha_AF7 alpha_SPF8 alpha_GF9];
y_pred = min_max_normalize_columns(probs', epsilon, 1)';

for i =1:N_test_NS0
    for j = 1:6
        if y_pred(i,j)==1
            count_raw(2,j) = count_raw(2,j)+1;
        end
    end
end
    end
    if k ==3

X_NS0_WF_validate = X_FF6_test*NS0_w1;
X_DF5_WF_validate = X_FF6_test*DF5_w1;
X_FF6_WF_validate = X_FF6_test*FF6_w1;
X_AF7_WF_validate = X_FF6_test*AF7_w1;
X_SPF8_WF_validate = X_FF6_test*SPF8_w1;
X_GF9_WF_validate = X_FF6_test*GF9_w1;

test_number = size(X_NS0_WF_validate,1);
test_number_range = test_number-10+1;
X_NS0_WF_validate_new = zeros(test_number_range,10);
X_DF5_WF_validate_new = zeros(test_number_range,10);
X_FF6_WF_validate_new = zeros(test_number_range,10);
X_AF7_WF_validate_new = zeros(test_number_range,10);
X_SPF8_WF_validate_new = zeros(test_number_range,10);
X_GF9_WF_validate_new = zeros(test_number_range,10);
for a = 1:test_number_range
    X_NS0_WF_validate_new(a,1:end) = X_NS0_WF_validate(a:a+9);
    X_DF5_WF_validate_new(a,1:end) = X_DF5_WF_validate(a:a+9);
    X_FF6_WF_validate_new(a,1:end) = X_FF6_WF_validate(a:a+9);
    X_AF7_WF_validate_new(a,1:end) = X_AF7_WF_validate(a:a+9);
    X_SPF8_WF_validate_new(a,1:end) = X_SPF8_WF_validate(a:a+9);
    X_GF9_WF_validate_new(a,1:end) = X_GF9_WF_validate(a:a+9); 
end
N_test_NS0 = size(X_NS0_WF_validate_new,1);
alpha_NS0 = zeros(N_test_NS0, 1); 
alpha_DF5 = zeros(N_test_NS0, 1);
alpha_FF6 = zeros(N_test_NS0, 1); 
alpha_AF7 = zeros(N_test_NS0, 1);
alpha_SPF8 = zeros(N_test_NS0, 1);
alpha_GF9 = zeros(N_test_NS0, 1); 
for b = 1:test_number_range
    logp_xn_given_zn_NS0 = Gmm_logp_xn_given_zn(X_NS0_WF_validate_new(b,1:end), phi_NS0);
    [~,~, loglik_NS0] = LogForwardBackward(logp_xn_given_zn_NS0, p_start_NS0, A_NS0);
    alpha_NS0(b, 1) = loglik_NS0;
    logp_xn_given_zn_DF5 = Gmm_logp_xn_given_zn(X_DF5_WF_validate_new(b,1:end), phi_DF5);
    [~,~, loglik_DF5] = LogForwardBackward(logp_xn_given_zn_DF5, p_start_DF5, A_DF5);
    alpha_DF5(b, 1) = loglik_DF5;
    logp_xn_given_zn_FF6 = Gmm_logp_xn_given_zn(X_FF6_WF_validate_new(b,1:end), phi_FF6);
    [~,~, loglik_FF6] = LogForwardBackward(logp_xn_given_zn_FF6, p_start_FF6, A_FF6);
    alpha_FF6(b, 1) = loglik_FF6;
    logp_xn_given_zn_AF7 = Gmm_logp_xn_given_zn(X_AF7_WF_validate_new(b,1:end), phi_AF7);
    [~,~, loglik_AF7] = LogForwardBackward(logp_xn_given_zn_AF7, p_start_AF7, A_AF7);
    alpha_AF7(b, 1) = loglik_AF7;
    logp_xn_given_zn_SPF8 = Gmm_logp_xn_given_zn(X_SPF8_WF_validate_new(b,1:end), phi_SPF8);
    [~,~, loglik_SPF8] = LogForwardBackward(logp_xn_given_zn_SPF8, p_start_SPF8, A_SPF8);
    alpha_SPF8(b, 1) = loglik_SPF8;
    logp_xn_given_zn_GF9 = Gmm_logp_xn_given_zn(X_GF9_WF_validate_new(b,1:end), phi_GF9);
    [~,~, loglik_GF9] = LogForwardBackward(logp_xn_given_zn_GF9, p_start_GF9, A_GF9);
    alpha_GF9(b, 1) = loglik_GF9;
end

probs = [alpha_NS0 alpha_DF5 alpha_FF6 alpha_AF7 alpha_SPF8 alpha_GF9];
y_pred = min_max_normalize_columns(probs', epsilon, 1)';

for i =1:N_test_NS0
    for j = 1:6
        if y_pred(i,j)==1
            count_raw(3,j) = count_raw(3,j)+1;
        end
    end
end
    end
    if k ==4

X_NS0_WF_validate = X_AF7_test*NS0_w1;
X_DF5_WF_validate = X_AF7_test*DF5_w1;
X_FF6_WF_validate = X_AF7_test*FF6_w1;
X_AF7_WF_validate = X_AF7_test*AF7_w1;
X_SPF8_WF_validate = X_AF7_test*SPF8_w1;
X_GF9_WF_validate = X_AF7_test*GF9_w1;
 
test_number = size(X_NS0_WF_validate,1);
test_number_range = test_number-10+1;
X_NS0_WF_validate_new = zeros(test_number_range,10);
X_DF5_WF_validate_new = zeros(test_number_range,10);
X_FF6_WF_validate_new = zeros(test_number_range,10);
X_AF7_WF_validate_new = zeros(test_number_range,10);
X_SPF8_WF_validate_new = zeros(test_number_range,10);
X_GF9_WF_validate_new = zeros(test_number_range,10);
for a = 1:test_number_range
    X_NS0_WF_validate_new(a,1:end) = X_NS0_WF_validate(a:a+9);
    X_DF5_WF_validate_new(a,1:end) = X_DF5_WF_validate(a:a+9);
    X_FF6_WF_validate_new(a,1:end) = X_FF6_WF_validate(a:a+9);
    X_AF7_WF_validate_new(a,1:end) = X_AF7_WF_validate(a:a+9);
    X_SPF8_WF_validate_new(a,1:end) = X_SPF8_WF_validate(a:a+9);
    X_GF9_WF_validate_new(a,1:end) = X_GF9_WF_validate(a:a+9); 
end
N_test_NS0 = size(X_NS0_WF_validate_new,1); 
alpha_NS0 = zeros(N_test_NS0, 1);
alpha_DF5 = zeros(N_test_NS0, 1); 
alpha_FF6 = zeros(N_test_NS0, 1); 
alpha_AF7 = zeros(N_test_NS0, 1);
alpha_SPF8 = zeros(N_test_NS0, 1); 
alpha_GF9 = zeros(N_test_NS0, 1); 
for b = 1:test_number_range
    logp_xn_given_zn_NS0 = Gmm_logp_xn_given_zn(X_NS0_WF_validate_new(b,1:end), phi_NS0);
    [~,~, loglik_NS0] = LogForwardBackward(logp_xn_given_zn_NS0, p_start_NS0, A_NS0);
    alpha_NS0(b, 1) = loglik_NS0;
    logp_xn_given_zn_DF5 = Gmm_logp_xn_given_zn(X_DF5_WF_validate_new(b,1:end), phi_DF5);
    [~,~, loglik_DF5] = LogForwardBackward(logp_xn_given_zn_DF5, p_start_DF5, A_DF5);
    alpha_DF5(b, 1) = loglik_DF5;
    logp_xn_given_zn_FF6 = Gmm_logp_xn_given_zn(X_FF6_WF_validate_new(b,1:end), phi_FF6);
    [~,~, loglik_FF6] = LogForwardBackward(logp_xn_given_zn_FF6, p_start_FF6, A_FF6);
    alpha_FF6(b, 1) = loglik_FF6;
    logp_xn_given_zn_AF7 = Gmm_logp_xn_given_zn(X_AF7_WF_validate_new(b,1:end), phi_AF7);
    [~,~, loglik_AF7] = LogForwardBackward(logp_xn_given_zn_AF7, p_start_AF7, A_AF7);
    alpha_AF7(b, 1) = loglik_AF7;
    logp_xn_given_zn_SPF8 = Gmm_logp_xn_given_zn(X_SPF8_WF_validate_new(b,1:end), phi_SPF8);
    [~,~, loglik_SPF8] = LogForwardBackward(logp_xn_given_zn_SPF8, p_start_SPF8, A_SPF8);
    alpha_SPF8(b, 1) = loglik_SPF8;
    logp_xn_given_zn_GF9 = Gmm_logp_xn_given_zn(X_GF9_WF_validate_new(b,1:end), phi_GF9);
    [~,~, loglik_GF9] = LogForwardBackward(logp_xn_given_zn_GF9, p_start_GF9, A_GF9);
    alpha_GF9(b, 1) = loglik_GF9;
end

probs = [alpha_NS0 alpha_DF5 alpha_FF6 alpha_AF7 alpha_SPF8 alpha_GF9];
y_pred = min_max_normalize_columns(probs', epsilon, 1)';

for i =1:N_test_NS0
    for j = 1:6
        if y_pred(i,j)==1
            count_raw(4,j) = count_raw(4,j)+1;
        end
    end
end
    end
    if k==5

X_NS0_WF_validate = X_SPF8_test*NS0_w1;
X_DF5_WF_validate = X_SPF8_test*DF5_w1;
X_FF6_WF_validate = X_SPF8_test*FF6_w1;
X_AF7_WF_validate = X_SPF8_test*AF7_w1;
X_SPF8_WF_validate = X_SPF8_test*SPF8_w1;
X_GF9_WF_validate = X_SPF8_test*GF9_w1;

test_number = size(X_NS0_WF_validate,1);
test_number_range = test_number-10+1;
X_NS0_WF_validate_new = zeros(test_number_range,10);
X_DF5_WF_validate_new = zeros(test_number_range,10);
X_FF6_WF_validate_new = zeros(test_number_range,10);
X_AF7_WF_validate_new = zeros(test_number_range,10);
X_SPF8_WF_validate_new = zeros(test_number_range,10);
X_GF9_WF_validate_new = zeros(test_number_range,10);
for a = 1:test_number_range
    X_NS0_WF_validate_new(a,1:end) = X_NS0_WF_validate(a:a+9);
    X_DF5_WF_validate_new(a,1:end) = X_DF5_WF_validate(a:a+9);
    X_FF6_WF_validate_new(a,1:end) = X_FF6_WF_validate(a:a+9);
    X_AF7_WF_validate_new(a,1:end) = X_AF7_WF_validate(a:a+9);
    X_SPF8_WF_validate_new(a,1:end) = X_SPF8_WF_validate(a:a+9);
    X_GF9_WF_validate_new(a,1:end) = X_GF9_WF_validate(a:a+9); 
end
N_test_NS0 = size(X_NS0_WF_validate_new,1);        
alpha_NS0 = zeros(N_test_NS0, 1);
alpha_DF5 = zeros(N_test_NS0, 1); 
alpha_FF6 = zeros(N_test_NS0, 1); 
alpha_AF7 = zeros(N_test_NS0, 1); 
alpha_SPF8 = zeros(N_test_NS0, 1); 
alpha_GF9 = zeros(N_test_NS0, 1); 
for b = 1:test_number_range
    logp_xn_given_zn_NS0 = Gmm_logp_xn_given_zn(X_NS0_WF_validate_new(b,1:end), phi_NS0);
    [~,~, loglik_NS0] = LogForwardBackward(logp_xn_given_zn_NS0, p_start_NS0, A_NS0);
    alpha_NS0(b, 1) = loglik_NS0;
    logp_xn_given_zn_DF5 = Gmm_logp_xn_given_zn(X_DF5_WF_validate_new(b,1:end), phi_DF5);
    [~,~, loglik_DF5] = LogForwardBackward(logp_xn_given_zn_DF5, p_start_DF5, A_DF5);
    alpha_DF5(b, 1) = loglik_DF5;
    logp_xn_given_zn_FF6 = Gmm_logp_xn_given_zn(X_FF6_WF_validate_new(b,1:end), phi_FF6);
    [~,~, loglik_FF6] = LogForwardBackward(logp_xn_given_zn_FF6, p_start_FF6, A_FF6);
    alpha_FF6(b, 1) = loglik_FF6;
    logp_xn_given_zn_AF7 = Gmm_logp_xn_given_zn(X_AF7_WF_validate_new(b,1:end), phi_AF7);
    [~,~, loglik_AF7] = LogForwardBackward(logp_xn_given_zn_AF7, p_start_AF7, A_AF7);
    alpha_AF7(b, 1) = loglik_AF7;
    logp_xn_given_zn_SPF8 = Gmm_logp_xn_given_zn(X_SPF8_WF_validate_new(b,1:end), phi_SPF8);
    [~,~, loglik_SPF8] = LogForwardBackward(logp_xn_given_zn_SPF8, p_start_SPF8, A_SPF8);
    alpha_SPF8(b, 1) = loglik_SPF8;
    logp_xn_given_zn_GF9 = Gmm_logp_xn_given_zn(X_GF9_WF_validate_new(b,1:end), phi_GF9);
    [~,~, loglik_GF9] = LogForwardBackward(logp_xn_given_zn_GF9, p_start_GF9, A_GF9);
    alpha_GF9(b, 1) = loglik_GF9;
end

probs = [alpha_NS0 alpha_DF5 alpha_FF6 alpha_AF7 alpha_SPF8 alpha_GF9];
y_pred = min_max_normalize_columns(probs', epsilon, 1)';

for i =1:N_test_NS0
    for j = 1:6
        if y_pred(i,j)==1
            count_raw(5,j) = count_raw(5,j)+1;
        end
    end
end
    end
    if k ==6

X_NS0_WF_validate = X_GF9_test*NS0_w1;
X_DF5_WF_validate = X_GF9_test*DF5_w1;
X_FF6_WF_validate = X_GF9_test*FF6_w1;
X_AF7_WF_validate = X_GF9_test*AF7_w1;
X_SPF8_WF_validate = X_GF9_test*SPF8_w1;
X_GF9_WF_validate = X_GF9_test*GF9_w1;

test_number = size(X_NS0_WF_validate,1);
test_number_range = test_number-10+1;
X_NS0_WF_validate_new = zeros(test_number_range,10);
X_DF5_WF_validate_new = zeros(test_number_range,10);
X_FF6_WF_validate_new = zeros(test_number_range,10);
X_AF7_WF_validate_new = zeros(test_number_range,10);
X_SPF8_WF_validate_new = zeros(test_number_range,10);
X_GF9_WF_validate_new = zeros(test_number_range,10);
for a = 1:test_number_range
    X_NS0_WF_validate_new(a,1:end) = X_NS0_WF_validate(a:a+9);
    X_DF5_WF_validate_new(a,1:end) = X_DF5_WF_validate(a:a+9);
    X_FF6_WF_validate_new(a,1:end) = X_FF6_WF_validate(a:a+9);
    X_AF7_WF_validate_new(a,1:end) = X_AF7_WF_validate(a:a+9);
    X_SPF8_WF_validate_new(a,1:end) = X_SPF8_WF_validate(a:a+9);
    X_GF9_WF_validate_new(a,1:end) = X_GF9_WF_validate(a:a+9); 
end
N_test_NS0 = size(X_NS0_WF_validate_new,1);      
alpha_NS0 = zeros(N_test_NS0, 1);
alpha_DF5 = zeros(N_test_NS0, 1); 
alpha_FF6 = zeros(N_test_NS0, 1); 
alpha_AF7 = zeros(N_test_NS0, 1); 
alpha_SPF8 = zeros(N_test_NS0, 1);
alpha_GF9 = zeros(N_test_NS0, 1); 
for b = 1:test_number_range
    logp_xn_given_zn_NS0 = Gmm_logp_xn_given_zn(X_NS0_WF_validate_new(b,1:end), phi_NS0);
    [~,~, loglik_NS0] = LogForwardBackward(logp_xn_given_zn_NS0, p_start_NS0, A_NS0);
    alpha_NS0(b, 1) = loglik_NS0;
    logp_xn_given_zn_DF5 = Gmm_logp_xn_given_zn(X_DF5_WF_validate_new(b,1:end), phi_DF5);
    [~,~, loglik_DF5] = LogForwardBackward(logp_xn_given_zn_DF5, p_start_DF5, A_DF5);
    alpha_DF5(b, 1) = loglik_DF5;
    logp_xn_given_zn_FF6 = Gmm_logp_xn_given_zn(X_FF6_WF_validate_new(b,1:end), phi_FF6);
    [~,~, loglik_FF6] = LogForwardBackward(logp_xn_given_zn_FF6, p_start_FF6, A_FF6);
    alpha_FF6(b, 1) = loglik_FF6;
    logp_xn_given_zn_AF7 = Gmm_logp_xn_given_zn(X_AF7_WF_validate_new(b,1:end), phi_AF7);
    [~,~, loglik_AF7] = LogForwardBackward(logp_xn_given_zn_AF7, p_start_AF7, A_AF7);
    alpha_AF7(b, 1) = loglik_AF7;
    logp_xn_given_zn_SPF8 = Gmm_logp_xn_given_zn(X_SPF8_WF_validate_new(b,1:end), phi_SPF8);
    [~,~, loglik_SPF8] = LogForwardBackward(logp_xn_given_zn_SPF8, p_start_SPF8, A_SPF8);
    alpha_SPF8(b, 1) = loglik_SPF8;
    logp_xn_given_zn_GF9 = Gmm_logp_xn_given_zn(X_GF9_WF_validate_new(b,1:end), phi_GF9);
    [~,~, loglik_GF9] = LogForwardBackward(logp_xn_given_zn_GF9, p_start_GF9, A_GF9);
    alpha_GF9(b, 1) = loglik_GF9;
end

probs = [alpha_NS0 alpha_DF5 alpha_FF6 alpha_AF7 alpha_SPF8 alpha_GF9];
y_pred = min_max_normalize_columns(probs', epsilon, 1)';

for i =1:N_test_NS0
    for j = 1:6
        if y_pred(i,j)==1
            count_raw(6,j) = count_raw(6,j)+1;
        end
    end
end
    end

    end

    varname = sprintf('count_raw%d', cishu);
    filename = sprintf('my_data_%s.txt', varname);
    dlmwrite(filename, count_raw, 'delimiter', '\t');
    
mat = count_raw;

acc = (mat(1,1)+mat(2,2)+mat(3,3)+mat(4,4)+mat(5,5)+mat(6,6))/(216*6)
acc_for(cishu) = acc;
   
weighted_cw = [NS0_w1';DF5_w1';FF6_w1';AF7_w1';SPF8_w1';GF9_w1'];
    varname = sprintf('weighted_cw%d', cishu);
    filename = sprintf('my_data_%s.txt', varname);
    dlmwrite(filename, weighted_cw, 'delimiter', '\t');
     
    NS0_w0 = NS0_w1;
    DF5_w0 = DF5_w1;
    FF6_w0 = FF6_w1;
    AF7_w0 = AF7_w1;
    SPF8_w0 = SPF8_w1;
    GF9_w0 = GF9_w1;
    disp(cishu);
end
toc
disp(['running time：',num2str(toc)]);

    varname = sprintf('acc_for%d', cishu);
    filename = sprintf('my_data_%s.txt', varname);
    dlmwrite(filename, acc_for, 'delimiter', '\t');
disp('complete');
