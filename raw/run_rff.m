% Usage:
%   matlab -batch "run_rff_raw"
% ── Load data (once per session) ──────────────────────────────────────────
load('GYdata.mat');
% ── Fixed parameters ──────────────────────────────────────────────────────
gamma         = 2;
vol_stdize    = 1;
num_sims      = 100;
% ── Demean settings ───────────────────────────────────────────────────────
demean_configs = [1, 1;
                  0, 0;
                  1, 0];
% ── Parameter grids ───────────────────────────────────────────────────────
windows       = [12, 60, 120];
zlist         = [0, logspace(0, 4, 5), 1e9];
% ── Prepare features (once per session) ───────────────────────────────────
X = [X, lagmatrix(Y, 1)];
% ── Start parallel pool (once per session) ────────────────────────────────
if isempty(gcp('nocreate'))
    parpool('local', feature('numcores'));
end
% ── Main loop ─────────────────────────────────────────────────────────────
SEP      = repmat('=', 1, 60);
t_global = tic;
for d = 1:size(demean_configs, 1)
    demeanX = demean_configs(d, 1);
    demeanY = demean_configs(d, 2);
for w = 1:numel(windows)
        trnwin = windows(w);
        param_str  = sprintf('gam%d_vol%d_win%d_dX%d_dY%d', ...
            gamma, vol_stdize, trnwin, demeanX, demeanY);
        output_dir = fullfile('raw_data', 'rff_fix', param_str);
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
        fprintf('  window=%d | demeanX=%d | demeanY=%d\n', trnwin, demeanX, demeanY);
parfor isim = 1:num_sims
            rff(X, Y, dates, zlist, gamma, vol_stdize, trnwin, demeanX, demeanY, isim, output_dir);
end
end  % window
end  % demean config
fprintf('\n%s\nRaw data rff done in %.2f min.\n%s\n', SEP, toc(t_global)/60, SEP);