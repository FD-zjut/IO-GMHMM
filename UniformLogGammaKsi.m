function [Gamma, Ksi] = UniformLogGammaKsi(LogGamma, LogKsi)
obj_num = length(LogGamma);
Q = size(LogGamma{1}, 2);
for q = 1:Q
    max_gamma_ary = zeros(1, obj_num);
    max_ksi_ary = zeros(1, obj_num);
    for r = 1:obj_num
        [Nr, Q] = size(LogGamma{r});
        max_gamma_ary(r) = max(LogGamma{r}(:,q));
        max_ksi_ary(r) = max(reshape(LogKsi{r}(:,q,:), (Nr-1)*Q, 1));
    end
    max_gamma = max(max_gamma_ary);
    max_ksi = max(max_ksi_ary);
    
    for r = 1:obj_num
        LogGamma{r}(:,q) = LogGamma{r}(:,q) - max_gamma;
        LogKsi{r}(:,q,:) = LogKsi{r}(:,q,:) - max_ksi;
    end
end

for r = 1:obj_num
    Gamma{r} = exp(LogGamma{r});
    Ksi{r} = exp(LogKsi{r});
end
end