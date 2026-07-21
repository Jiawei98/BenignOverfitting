% run_simple.m
% Runs simple OLS prediction (no RFF) for raw data.
% Called once per node by SLURM — no repeated MATLAB launches.
%
% Usage:
%   matlab -batch "run_simple_raw"

% ── Load data (once per session) ──────────────────────────────────────────
load('GYdata.mat');

% ── Fixed parameters ──────────────────────────────────────────────────────
vol_stdize    = 1;

% ── Parameter grids ───────────────────────────────────────────────────────
windows = [12, 60, 120];

% Each row = [demeanX, demeanY]
demean_settings = [0, 0;   % setting 1
                   1, 1;   % setting 2
                   1, 0];  % setting 3

% ── Prepare features (once per session) ───────────────────────────────────
X = [X, lagmatrix(Y, 1)];

% ── Main loop ─────────────────────────────────────────────────────────────
SEP      = repmat('=', 1, 60);
t_global = tic;

for w = 1:numel(windows)
    trnwin = windows(w);

    for m = 1:size(demean_settings, 1)
        demeanX = demean_settings(m, 1);
        demeanY = demean_settings(m, 2);

        param_str  = sprintf('vol%d_win%d_dX%d_dY%d', ...
            vol_stdize, trnwin, demeanX, demeanY);
        output_dir = fullfile('raw_data', 'simple_fix', param_str);
        if ~exist(output_dir, 'dir'), mkdir(output_dir); end

        fprintf('  window=%d | demeanX=%d | demeanY=%d\n', trnwin, demeanX, demeanY);
        simple(X, Y, dates, vol_stdize, trnwin, demeanX, demeanY, output_dir);
    end  % demean setting
end  % window

fprintf('\n%s\nAll raw data simple model done in %.2f min.\n%s\n', ...
        SEP, toc(t_global)/60, SEP);