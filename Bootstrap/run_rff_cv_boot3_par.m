%% run_rff_cv_boot3_par.m
load('GYdata.mat');
gamma      = 2;
vol_stdize = 1;
num_sims   = 100;
num_boots  = 100;
demeanX    = 1;
demeanY    = 1;
windows    = [12, 60, 120];
nval       = 1;
zlist      = logspace(-4, 9, 28);
X = [X, lagmatrix(Y, 1)];
SEP      = repmat('=', 1, 60);
% ── Open parallel pool ────────────────────────────────────────────────────
if isempty(gcp('nocreate'))
    parpool('local');
end
% ─────────────────────────────────────────────────────────────────────────
t_global = tic;
for w = 1:numel(windows)
    trnwin = windows(w);
    fprintf('\n%s\nwindow=%d | nval=%d | demeanX=%d | demeanY=%d\n%s\n', ...
        SEP, trnwin, nval, demeanX, demeanY, SEP);
    for iboot = 0:num_boots
        t_boot = tic;
        param_str  = sprintf('gam%d_vol%d_win%d_nval%d_dX%d_dY%d', ...
            gamma, vol_stdize, trnwin, nval, demeanX, demeanY);
        output_dir = fullfile('raw_data', 'rff_cv_boot3', param_str, sprintf('boot%d', iboot));
        if ~exist(output_dir, 'dir'), mkdir(output_dir); end
        % ── Broadcast variables needed inside parfor ──────────────────
        iboot_bc      = iboot;
        output_dir_bc = output_dir;
        % ─────────────────────────────────────────────────────────────
        parfor isim = 1:num_sims
            t_sim = tic;
            rff_cv_boot3(X, Y, dates, zlist, gamma, vol_stdize, trnwin, nval, ...
                demeanX, demeanY, isim, output_dir_bc, iboot_bc);
            fprintf('  boot %3d/%d | sim %3d/%d | %.2f sec\n', ...
                iboot_bc, num_boots, isim, num_sims, toc(t_sim));
        end
        fprintf('Boot %d done in %.2f min.\n', iboot, toc(t_boot)/60);
    end
    fprintf('\n%s\nwindow=%d done in %.2f min.\n%s\n', SEP, trnwin, toc(t_global)/60, SEP);
end
fprintf('\n%s\nAll boots done in %.2f min.\n%s\n', SEP, toc(t_global)/60, SEP);