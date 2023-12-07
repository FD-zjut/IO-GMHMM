function [prior, mu, Sigma, loglik] = Gmm(X, mix_num, varargin)

for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'cov_type'
            cov_type = varargin{i1+1};
        case 'cov_thresh'
            cov_thresh = varargin{i1+1};
        case 'restart_num'
            restart_num = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
    end
end
if (~exist('cov_type'))
    cov_type = 'diag';     
end
if (~exist('cov_thresh'))
    cov_thresh = 1e-4;     
end
if (~exist('restart_num'))
    restart_num = 1;     
end
if (~exist('iter_num'))
    iter_num = 100;   
end

[pi0, mu0, Sigma0] = Gmm_init_by_kmeans(X, mix_num, restart_num, cov_thresh);

[prior, mu, Sigma, loglik] = Gmm_em(X, pi0, mu0, Sigma0, iter_num, cov_type, cov_thresh);

end

function [pi0, mu0, Sigma0] = Gmm_init_by_kmeans(X, mix_num, restart_num, cov_thresh)

[N,p] = size(X);
err = inf;
for i1 = 1:restart_num
    [indi_curr, mu0_curr, errs] = kmeans( X, mix_num );
    err_curr = sum( errs );
    mu0_curr = mu0_curr';
    if err_curr < err
        err = err_curr;
        mu0 = mu0_curr;
        indi = indi_curr;
    end
end

pi0 = zeros(1,mix_num);
for i1 = 1:mix_num
    pi0(i1) = sum(indi==i1);
end
pi0 = pi0 / length(indi);

for i1 = 1:mix_num
    X_curr = X(indi==i1, :);
    mu0_curr = mu0(:,i1);
    Sigma0_curr = cov(bsxfun(@minus, X_curr, mu0_curr'));

    if min(eig(Sigma0_curr)) < cov_thresh   
        Sigma0(:,:,i1) = Sigma0_curr + cov_thresh * eye(p);
    else
        Sigma0(:,:,i1) = Sigma0_curr;  
    end
end
end

function [prior, mu, Sigma, loglik] = Gmm_em(X, prior, mu, Sigma, iter_num, cov_type, cov_thresh)
M = length(prior); 
[N,p] = size(X);
pre_ll = -inf;

for k = 1:iter_num    

    p_xn_given_zn = zeros(N, M);
    for i1 = 1:M
        p_xn_given_zn(:, i1) = mvnpdf(X, mu(:,i1)', Sigma(:,:,i1));
    end

    numer = bsxfun(@times, p_xn_given_zn, prior);    
    denor = sum(numer, 2);                 
    gamma = bsxfun(@rdivide, numer, denor); 

    Nk = sum(gamma, 1);    
    mu = bsxfun(@rdivide, (X' * gamma), Nk);

    for i1 = 1:M
        x_minus_mu = bsxfun(@minus, X, mu(:,i1)');
        Sigma(:,:,i1) = bsxfun(@times, gamma(:,i1), x_minus_mu)' * x_minus_mu / Nk(i1);
        if (cov_type=='diag')
            Sigma(:,:,i1) = diag(diag(Sigma(:,:,i1)));
        end
        if min(eig(Sigma(:,:,i1))) < cov_thresh    
            Sigma(:,:,i1) = Sigma(:,:,i1) + cov_thresh * eye(p);
        end
        prior = Nk / N;
    end
    
    p_xn = zeros(N,1);
    for i1 = 1:M
        p_xn = p_xn + prior(i1) * mvnpdf(X, mu(:,i1)', Sigma(:,:,i1));
    end
    loglik = sum(log(p_xn));
    
    if (loglik-pre_ll<log(1.0001)) break;
    else pre_ll = loglik; end
    
end

end

function DebugPlot(X, gamma, mu, Sigma)
    M = size(mu,2);
    scatter(X(:,1), X(:,2), [], gamma), hold on, axis([-2 6 -1.5 3.5])
    pause
    for i1 = 1:M
        error_ellipse(Sigma(i1,:,:), mu(i1,:), 'style', 'r', 'conf', 0.9)
    end
    hold off
    pause
end