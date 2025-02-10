function [BinSeq, psl_history, epoch] = LSEFalgorithm(m, n, t, h, G, maxIteration, report)
    % Initialize
    BinSeq = 2 * (rand(m, n) > 0.5) - 1;  % Random binary sequence
    Omega = mimoxcorr(BinSeq);
    BestCost = lsefitnessmimo(Omega.', t);
    % Cost = 0;

    isGImpr = true;
    isLImpr = false;

    % Auxiliary variables
    epoch = 0;
    epoch_NB = 0;
    reverseStr = '';

    while (epoch < maxIteration)
        epoch = epoch + 1;
        if (isGImpr)
            idx_perm_n = randperm(n);
            idx_perm_m = randperm(m);

            for j = 1:m
                for i = 1:n
                    pos_n = idx_perm_n(i);
                    pos_m = idx_perm_m(j);
                    
                    % Flip bit at position pos and calculate new correlation
                    Omega_pos = Neighborsmmimoxcorr(BinSeq.', Omega.', pos_n, pos_m);
                    
                    [Cost, psl] = lsefitnessmimo(Omega_pos, t);
                    
                    epoch_NB = epoch_NB + 1;
                    psl_history(epoch_NB) = psl;
    
                    if abs(psl - G) < 1e-5
                        BinSeq(pos_m, pos_n) = -BinSeq(pos_m, pos_n);
                        if report
                            fprintf("\n");
                        end
                        return;
                    end
                    % Update if better solution found
                    if Cost < BestCost
                        BestCost = Cost;
                        isLImpr = true;
                        BinSeq(pos_m, pos_n) = -BinSeq(pos_m, pos_n);
                        Omega = Omega_pos.';
                        break;
                    end
                end
            end
        
            if isLImpr
                isGImpr = true;
                isLImpr = false;
                continue;
            else
                isGImpr = false;
            end
        else

            r_n = randi([1, n], 1, h);
            r_m = randi([1, m], 1, h);

            for ii = 1:length(r_n)
                Omega_pos = Neighborsmmimoxcorr(BinSeq.', Omega.', r_n(ii), r_m(ii));
                Omega = Omega_pos.';
                BinSeq(r_m(ii), r_n(ii)) = -BinSeq(r_m(ii), r_n(ii));
            end
            BestCost = lsefitnessmimo(Omega_pos, t);
            
            isGImpr = true;
            isLImpr = false;
        end

        if report
            msg = sprintf('NewSHCMIMO, epoch:%d, psl = %.0f', epoch, psl);
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end
    end

    if report
        fprintf("\n");
    end
end
function Omega = mimoxcorr(BinSeq)
    [m, n] = size(BinSeq);
    Omega = zeros(m^2,2*n-1);
    % BinSeq mxn
    for i = 1:m
        for j = 1:m
            Omega((i-1)*m+j, :) = xcorr(BinSeq(i,:), BinSeq(j,:));
        end
    end
end

function [fit, psl] = lsefitnessmimo(corr, t)
    [n_2, m_2] = size(corr); 
    m = sqrt(m_2);
    n = round((n_2+1)/2); 
    corr_abs = abs(reshape(corr(1:n-1,:), [], 1));
    psl = max(corr_abs);
    fit = lset((corr_abs)/n,t/n);
end

function lse = lset(x, t)
    [~, n] = size(x);
    lse = zeros(n, 1);
    for i = 1:n
        max_x = max(x(:, i));
        lse(i) = t * log(sum(exp((x(:, i) - max_x) / t))) + max_x;
    end
end

function r_original = Neighborsxcorr_s(x, y, r_original, idx_x, idx_y)
    N = length(x);
    if ~isempty(idx_y)
        % 预分配矩阵
        A = zeros(2*N-1, length(idx_y));
        
        % 构建矩阵A的高效方法
        for i = 1:length(idx_y)
            start_idx = N-(idx_y(i)-1);
            A(start_idx:start_idx+N-1, i) = x;
        end
    
        % B = conj(flip(A));
    
        r_original = r_original - 2*A.*repmat(y(idx_y,1)', 2*N-1, 1);
    end

    if ~isempty(idx_x)
        % 预分配矩阵
        A = zeros(2*N-1, length(idx_x));
        
        % 构建矩阵A的高效方法
        for i = 1:length(idx_x)
            start_idx = N-(idx_x(i)-1);
            A(start_idx:start_idx+N-1, i) = y;
        end
    
        B = conj(flip(A));
    
        r_original = r_original - 2*B.*repmat(x(idx_x,1)', 2*N-1, 1);
    end

end

function Omega_T_new = Neighborsmmimoxcorr(BinSeq_T, Omega_T, idx_n, idx_m)
    [n_2, m_2] = size(Omega_T); 
    m = sqrt(m_2);
    n = (n_2+1)/2; 
    listidx = 1:m;
    listidx(idx_m) = [];

    % BinSeq_T(idx_n, idx_m) = -BinSeq_T(idx_n, idx_m);

    for i = listidx
        Omega_T(:, (i-1)*m+idx_m) = Neighborsxcorr_s(BinSeq_T(:,i), BinSeq_T(:,idx_m), Omega_T(:, (i-1)*m+idx_m), [], idx_n);

        % BinSeq_T(idx_n, idx_m) = -BinSeq_T(idx_n, idx_m);
        % Omega_T(:, (i-1)*m+idx_m) = xcorr(BinSeq_T(:,i), BinSeq_T(:,idx_m));
        % r_corr = Omega_T(:, (i-1)*m+idx_m);
    end

    for j = listidx 
        % Omega_T(:, (idx_m-1)*m+j) = xcorr(BinSeq_T(:,idx_m), BinSeq_T(:,j));
        Omega_T(:, (idx_m-1)*m+j) = Neighborsxcorr_s(BinSeq_T(:,idx_m), BinSeq_T(:,j), Omega_T(:, (idx_m-1)*m+j), idx_n, []);
    end

    % Omega_T(:, (idx_m-1)*m+idx_m) = xcorr(BinSeq_T(:,idx_m), BinSeq_T(:,idx_m));
    Omega_T(:, (idx_m-1)*m+idx_m) = Neighborsxcorr_s(BinSeq_T(:,idx_m), BinSeq_T(:,idx_m), Omega_T(:, (idx_m-1)*m+idx_m), idx_n, idx_n);

    Omega_T_new = Omega_T;

    % N = length(x);
    % A = zeros(2*N-1, length(idx));
    % for i = 1:length(idx)
    %     start_idx = N-(idx(i)-1);
    %     A(start_idx:start_idx+N-1, i) = x;
    % end
    % 
    % B = conj(flip(A));
    % 
    % r_new = r_original - 2*A.*repmat(x(idx,1)', 2*N-1, 1) - 2*B.*repmat(x(idx,1).', 2*N-1, 1);
end