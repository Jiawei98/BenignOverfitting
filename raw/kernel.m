% kernel.m
%
% Implements the six timing strategies from Table III and computes
% SR, Corr (vs History), and alpha t-stats (t_a1, t_a2),
% reported separately for three sub-periods.
%
% Strategies (notation follows the table note):
%
%   History  : mu_hat = (1/trnwin) * sum_{k=0}^{trnwin-1} R_{t-k}
%   TsMom    : mu_hat = (1/sigma_y,t) * (1/trnwin) * sum R_{t-k}
%                       (vol-normalised EW = Sharpe-ratio timing signal)
%   VolMom   : mu_hat = 0.05 * (1/sigma2_x,t) * sum_{k=0}^{trnwin-1} (trnwin-k)/W * R_{t-k}
%   DW       : mu_hat = sum_{k=0}^{trnwin-1} (trnwin-k)/W * R_{t-k}
%   PV       : mu_hat = 1 / sigma2_x,t
%   EWPV     : mu_hat = (1/sigma2_x,t) * (1/trnwin) * sum R_{t-k}
%
%   sigma_y,t   = rolling std of Y over the training window
%   sigma2_x,t  = average cross-predictor variance over the training window
%   W           = trnwin*(trnwin+1)/2  (DW normalisation so weights sum to 1)
%
%   t_a1 : managed_port ~ const + market_ret
%   t_a2 : managed_port ~ const + market_ret + hist_managed_port

clear; clc;

% ===========================================================================
%  PARAMETERS
% ===========================================================================
vol_stdize  = 1;
demean      = 0;

% Sub-periods
subbeg      = [1930, 1975, 1930];
subend      = [1974, 2020, 2020];
panel_names = {'Panel A: 1930-1974', 'Panel B: 1975-2020', 'Panel C: 1930-2020'};
n_sp        = length(subbeg);

% ===========================================================================
%  LOAD AND PREPROCESS DATA
%  Matches the convention in rff.m / simple.m exactly.
% ===========================================================================
fprintf('Loading GYdata.mat ...\n');
load('GYdata.mat');

X = [X, lagmatrix(Y, 1)];

if vol_stdize == 1
    X  = volstdbwd(X, []);
    Y2 = movmean(Y.^2, [11 0]);
    Y  = Y(37:end) ./ sqrt(Y2(36:end-1));
    X  = X(37:end, :);
    dates = dates(37:end, :);
end

T_full = length(Y);
X      = X';       % (d x T)
Y      = Y';       % (1 x T)
d      = size(X, 1);

fprintf('  d = %d features  |  T = %d\n', d, T_full);

% Extract calendar years for sub-period filtering
if isnumeric(dates)
    years_full = floor(dates(:, 1) / 100);   % YYYYMM -> YYYY
else
    years_full = str2double(cellstr(num2str(dates(:, 1:4))));
end

% Output folder
out_root = fullfile('tables', 'kernel_NW4');
if ~exist(out_root, 'dir'), mkdir(out_root); end

% ===========================================================================
%  LOOP OVER WINDOWS
% ===========================================================================
for trnwin = [12]

W_dw        = trnwin * (trnwin + 1) / 2;
weights_EW  = ones(1, trnwin) / trnwin;
weights_DW  = (trnwin - (0:trnwin-1)) / W_dw;   % k=0 -> highest weight

% ===========================================================================
%  STORAGE
% ===========================================================================
yhat_hist   = nan(1, T_full);
yhat_TsMom  = nan(1, T_full);
yhat_VolMom = nan(1, T_full);
yhat_DW     = nan(1, T_full);
yhat_PV     = nan(1, T_full);
yhat_EWPV   = nan(1, T_full);

% ===========================================================================
%  MAIN LOOP
% ===========================================================================
for t = (trnwin + 1) : T_full

    % ── Slice training window ─────────────────────────────────────────────
    trnloc = (t - trnwin) : (t - 1);
    Ztrn   = X(:, trnloc);     % (d x trnwin) predictors
    Ytrn   = Y(trnloc);        % (1 x trnwin) returns

    if demean
        Ymn = nanmean(Ytrn);
        Zmn = nanmean(Ztrn, 2);
    else
        Ymn = 0;
        Zmn = zeros(d, 1);
    end

    Ytrn = Ytrn - Ymn;
    Ztrn = Ztrn - Zmn;

    % Reverse so index 1 = most recent (preserves DW weight ordering)
    Yvec = fliplr(Ytrn);       % Yvec(1)=R_{t-1}, Yvec(2)=R_{t-2}, ...

    % ── Scalar quantities ─────────────────────────────────────────────────
    sigma_y  = nanstd(Ytrn);            % rolling Y vol (scalar)
    sigma2_x = mean(var(Ztrn, 0, 2));   % avg predictor variance (scalar)

    % ── Forecasts ─────────────────────────────────────────────────────────
    yhat_hist(t)   = weights_EW * Yvec' + Ymn;
    yhat_TsMom(t)  = (1 / sigma_y)   * (weights_EW * Yvec') + Ymn;
    yhat_DW(t)     = weights_DW * Yvec' + Ymn;
    yhat_PV(t)     = 1 / sigma2_x;
    yhat_VolMom(t) = 0.05 * (1 / sigma2_x) * (weights_DW * Yvec') + Ymn;
    yhat_EWPV(t)   = (1 / sigma2_x) * (weights_EW * Yvec') + Ymn;

end

% ===========================================================================
%  EVALUATION SLICE
% ===========================================================================
eval_idx   = (trnwin + 1) : T_full;
Y_eval     = Y(eval_idx)';           % (n x 1)
years_eval = years_full(eval_idx);   % calendar years aligned to eval_idx

yhats = struct( ...
    'History', yhat_hist(eval_idx)',    ...
    'TsMom',   yhat_TsMom(eval_idx)',   ...
    'VolMom',  yhat_VolMom(eval_idx)',  ...
    'DW',      yhat_DW(eval_idx)',      ...
    'PV',      yhat_PV(eval_idx)',      ...
    'EWPV',    yhat_EWPV(eval_idx)');

model_names = fieldnames(yhats);
n_models    = numel(model_names);

% Gross managed portfolio returns: port_t = mu_hat_t * R_t
port = structfun(@(w) w .* Y_eval, yhats, 'UniformOutput', false);

% ===========================================================================
%  METRICS BY SUB-PERIOD
% ===========================================================================
freq = 12;
SEP  = repmat('=', 1, 54);

% Storage for saving
R2_mat   = nan(n_models, n_sp);
SR_mat   = nan(n_models, n_sp);
Corr_mat = nan(n_models, n_sp);
Ta1_mat  = nan(n_models, n_sp);
Ta2_mat  = nan(n_models, n_sp);

for sp = 1:n_sp
    beg_yr  = subbeg(sp);
    end_yr  = subend(sp);
    sp_mask = years_eval >= beg_yr & years_eval <= end_yr;

    if sum(sp_mask) == 0
        fprintf('[%d-%d] No data — skipping.\n', beg_yr, end_yr);
        continue
    end

    Y_sp          = Y_eval(sp_mask);
    port_hist_sp  = port.History(sp_mask);
    yhat_hist_sp  = yhats.History(sp_mask);   % raw predictions for R2 benchmark

    fprintf('\n%s\n%s  (n = %d, trnwin = %d)\n%s\n', ...
            SEP, panel_names{sp}, sum(sp_mask), trnwin, SEP);
    fprintf('%-10s  %8s  %6s  %6s  %8s  %8s\n', 'Portfolio','OOS_R2','SR','Corr','t_a1','t_a2');
    fprintf('%s\n', repmat('-', 1, 46));

    for k = 1:n_models
        name  = model_names{k};
        pret  = port.(name)(sp_mask);
        valid = ~isnan(pret);

        % Sharpe ratio (annualised)
        sr = mean(pret(valid)) / std(pret(valid)) * sqrt(freq);

        % Corr with History managed portfolio
        if strcmp(name, 'History')
            cval = 1.00;
        else
            cc   = corrcoef(pret(valid), port_hist_sp(valid));
            cval = cc(1, 2);
        end

        % t_a1: port ~ const + mkt
        lags      = 4; %nw_lag_select(sum(valid));
        [a1, ta1] = reg_alpha(pret(valid), Y_sp(valid), lags);

        % t_a2: port ~ const + mkt + hist_port (undefined for History)
        if strcmp(name, 'History')
            ta2 = NaN;
        else
            [~, ta2] = reg_alpha(pret(valid), ...
                                 [Y_sp(valid), port_hist_sp(valid)], lags);
        end

        % OOS R2 vs history-average benchmark
        if strcmp(name, 'History')
            r2val = 0;
        else
            yhat_k = yhats.(name)(sp_mask);
            r2val  = 1 - sum((Y_sp(valid) - yhat_k(valid)).^2) / ...
                         sum((Y_sp(valid) - yhat_hist_sp(valid)).^2);
        end

        % Store
        R2_mat(k, sp)   = r2val;
        SR_mat(k, sp)   = sr;
        Corr_mat(k, sp) = cval;
        Ta1_mat(k, sp)  = ta1;
        Ta2_mat(k, sp)  = ta2;

        ta2_str = '';
        if ~isnan(ta2), ta2_str = sprintf('%8.3f', ta2); end
        fprintf('%-10s  %8.4f  %6.3f  %6.3f  %8.3f  %s\n', name, r2val, sr, cval, ta1, ta2_str);
    end

    % --- Diagnostic Plot for PV Strategy ---
    figure('Name', sprintf('PV Diagnostics Win=%d', trnwin));
    subplot(2,1,1);
    plot(dates(eval_idx), yhats.PV, 'r', 'LineWidth', 1.5); hold on;
    plot(dates(eval_idx), yhats.History, 'k--', 'LineWidth', 1);
    title('Forecast Scale: PV vs History');
    legend('PV Forecast (1/sigma2_x)', 'History Avg');
    ylabel('Forecast Value');

    subplot(2,1,2);
    % Cumulative returns of the managed portfolio
    plot(dates(eval_idx), cumprod(1 + port.PV), 'r'); hold on;
    plot(dates(eval_idx), cumprod(1 + port.History), 'k--');
    title('Cumulative Performance');
    legend('PV Managed Portfolio', 'History Managed Portfolio');
end

% ===========================================================================
%  SAVE TABLE
% ===========================================================================
% Build a flat table: one row per (model x sub-period)
panel_labels = arrayfun(@(b,e) sprintf('%d-%d', b, e), ...
                        subbeg, subend, 'UniformOutput', false);

rows_Model  = {};
rows_Period = {};
rows_R2     = [];
rows_SR     = [];
rows_Corr   = [];
rows_Ta1    = [];
rows_Ta2    = [];

for sp = 1:n_sp
    for k = 1:n_models
        rows_Model{end+1}  = model_names{k};   %#ok<AGROW>
        rows_Period{end+1} = panel_labels{sp};  %#ok<AGROW>
        rows_R2(end+1)     = R2_mat(k, sp);     %#ok<AGROW>
        rows_SR(end+1)     = SR_mat(k, sp);     %#ok<AGROW>
        rows_Corr(end+1)   = Corr_mat(k, sp);   %#ok<AGROW>
        rows_Ta1(end+1)    = Ta1_mat(k, sp);    %#ok<AGROW>
        rows_Ta2(end+1)    = Ta2_mat(k, sp);    %#ok<AGROW>
    end
end

T_out = table(rows_Model(:), rows_Period(:), ...
              rows_R2(:), rows_SR(:), rows_Corr(:), rows_Ta1(:), rows_Ta2(:), ...
              'VariableNames', {'Model','Period','OOS_R2','SR','Corr','t_a1','t_a2'});

writetable(T_out, fullfile(out_root, sprintf('win%d.csv', trnwin)));
fprintf('\n  Saved: %s\n', fullfile(out_root, sprintf('win%d.csv', trnwin)));

end  % trnwin loop

% ===========================================================================
%  LOCAL HELPERS
% ===========================================================================

function L = nw_lag_select(T)
% Newey-West automatic lag: floor(4*(T/100)^(2/9)), minimum 1
% Reference: Newey & West (1994), Review of Economic Studies, 61(4), 631-653.
    L = max(1, floor(4 * (T / 100)^(2/9)));
end

function S = nw_meat(X, e, L)
% Newey-West HAC "meat" matrix
%   S = sum_t e_t^2*x_t*x_t'  +  sum_{l=1}^{L} w_l*(Gamma_l + Gamma_l')
%   w_l = 1 - l/(L+1)
    Xe = X .* e;
    S  = Xe' * Xe;
    for l = 1:L
        Gl = Xe(l+1:end,:)' * Xe(1:end-l,:);
        S  = S + (1 - l/(L+1)) * (Gl + Gl');
    end
end

function [alpha, t_alpha] = reg_alpha(y, X_reg, nlags)
% OLS regression  y = alpha + X_reg*beta + e
% alpha    : OLS intercept
% t_alpha  : Newey-West HAC t-statistic for H0: alpha = 0
    T      = length(y);
    Xc     = [ones(T, 1), X_reg];
    b      = (Xc' * Xc) \ (Xc' * y);
    e      = y - Xc * b;
    S      = nw_meat(Xc, e, nlags);
    XtX_i  = (Xc' * Xc) \ eye(size(Xc, 2));
    V      = XtX_i * S * XtX_i;
    alpha   = b(1);
    t_alpha = b(1) / sqrt(max(V(1, 1), 0));
end