function [p_start, A] = M_step_common(Gamma, Ksi)
    obj_num = length(Gamma);
    Q = size(Gamma{1},2);
    
    p_start_numer = zeros(1,Q);
    for r = 1:obj_num
        p_start_numer = p_start_numer + Gamma{r}(1,:);
    end
    p_start = p_start_numer / sum(p_start_numer);
    
    A_numer = zeros(Q,Q);
    for r = 1:obj_num
        A_numer = A_numer + reshape(sum(Ksi{r},1), Q, Q);
    end
    A = bsxfun(@rdivide, A_numer, sum(A_numer,2));
end