function [loggamma, logksi, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A)

    [N,Q] = size(logp_xn_given_zn);
    logalpha = zeros(N,Q);
    logbeta = zeros(N,Q);
    logc = zeros(N,1);
    loggamma = zeros(N,Q);
    logksi = zeros(N,Q,Q);

    tmp = logp_xn_given_zn(1,:) + log(p_start);
    logc(1) = log( sum( exp( tmp - max(tmp) ) ) ) + max(tmp);
    logalpha(1,:) = -logc(1) + logp_xn_given_zn(1,:) + log(p_start);
    logbeta(N,:) = 0;

    for n = 2:N
        tmp = bsxfun(@plus, bsxfun(@plus, log(A), logalpha(n-1,:)'), logp_xn_given_zn(n,:));
        logc(n) = log ( sum( sum ( exp ( tmp - max(tmp(:)) ) ) ) ) + max(tmp(:));
        for q = 1:Q
            tmp2 = logalpha(n-1,:) + log(A(:,q)');
            if (isinf(max(tmp2)))
                logalpha(n,q) = -inf;
%                 logalpha(n,q) = -logc(n) + logp_xn_given_zn(n,q) + log( sum( exp( tmp2 - max(tmp2) ) ) ) + max(tmp2);
            else
                logalpha(n,q) = -logc(n) + logp_xn_given_zn(n,q) + log( sum( exp( tmp2 - max(tmp2) ) ) ) + max(tmp2);
            end
        end
    end

    for n = N-1:-1:1
        for q = 1:Q
            tmp = logbeta(n+1,:) + logp_xn_given_zn(n+1,:) + log(A(q,:));
            logbeta(n,q) = -logc(n+1) + log( sum( exp( tmp - max(tmp) ) ) ) + max(tmp);
        end
    end

    loggamma = logalpha + logbeta;
   
    for n = 2:N
        logksi(n,:,:) = -logc(n) + bsxfun(@plus, bsxfun(@plus, log(A), logalpha(n-1,:)'), logp_xn_given_zn(n,:) + logbeta(n,:));
    end
    logksi(1,:,:) = [];

    loglik = sum(logc);
end