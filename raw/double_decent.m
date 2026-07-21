%% double_descent.m
% Test risk only, ridgeless (z=0) + z=1e9 dashed, all 3 windows overlaid.
% X-axis: gamma = P/n, log scale.

% ── Configuration ────────────────────────────────────────────────────────
gamma_param = 2;
vol_stdize  = 1;
num_z       = 7;
demean      = 1;

windows  = [12, 60, 120];
rff_root = fullfile('raw_data', 'rff_fix');
out_dir  = fullfile('plots', 'double_descent');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ── z grid (must match run_rff_raw.m) ────────────────────────────────────
zlist = [0, logspace(0, 4, 5), 1e9];   % z=0 → ridgeless; z=1e9 → history avg

% ── Feature grid ─────────────────────────────────────────────────────────
maxP  = 12000;
Plist = round(logspace(log10(2), log10(maxP), 100));
nP    = length(Plist);

% ── One color per window ─────────────────────────────────────────────────
win_colors = [
    0.00, 0.45, 0.74;   % blue  – win = 12
    0.85, 0.33, 0.10;   % red   – win = 60
    0.47, 0.67, 0.19;   % green – win = 120
];
win_labels = {'$T = 12$', '$T = 60$', '$T = 120$'};

% ── Figure ───────────────────────────────────────────────────────────────
f       = figure('Color', 'w', 'Position', [100, 100, 680, 480], 'Visible', 'off');
ax_risk = axes('Parent', f);
hold(ax_risk, 'on');

h_legend = gobjects(numel(windows), 1);   % one handle per window for legend

for w = 1:numel(windows)
    trnwin = windows(w);
    col    = win_colors(w, :);

    % ── Load Sim1 ────────────────────────────────────────────────────────
    param_str = sprintf('gam%d_vol%d_win%d_dX%d_dY%d', ...
                        gamma_param, vol_stdize, trnwin, demean, demean);
    sim_file  = fullfile(rff_root, param_str, 'Sim1.mat');
    if ~exist(sim_file, 'file')
        fprintf('Sim1 not found for win=%d, skipping.\n', trnwin);
        continue;
    end
    d        = load(sim_file);
    Y_actual = d.Y(:);
    T_len    = length(Y_actual);
    eval_idx = (trnwin + 1):T_len;

    % ── Test MSE ─────────────────────────────────────────────────────────
    Yprd        = d.Yprd;                              % [T x nP x nZ]
    Y_rep       = repmat(Y_actual, [1, nP, num_z]);
    TestMSE     = (Yprd - Y_rep).^2;
    avg_TestMSE = squeeze(nanmean(TestMSE(eval_idx, :, :), 1));  % [nP x nZ]

    % ── X-axis: gamma = P / n, sorted ascending ──────────────────────────
    gamma_x       = Plist ./ trnwin;
    [gx_sort, ix] = sort(gamma_x);
    mse_sort      = avg_TestMSE(ix, :);

    % ── Ridgeless (z = 0, col 1): solid line with dots ───────────────────
    h_legend(w) = plot(ax_risk, gx_sort, mse_sort(:, 1), '-o', ...
        'LineWidth', 2.0, 'Color', col, 'MarkerSize', 3.5, ...
        'MarkerFaceColor', col, 'DisplayName', win_labels{w});

    % ── History avg (z = 1e9, last col): thin dashed, no markers ─────────
    plot(ax_risk, gx_sort, mse_sort(:, end), '--', ...
        'LineWidth', 1.2, 'Color', col, 'HandleVisibility', 'off');
end

% ── Vertical reference line at gamma = 1 ─────────────────────────────────
xline(ax_risk, 1, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, ...
      'HandleVisibility', 'off');

% ── Axes formatting ───────────────────────────────────────────────────────
set(ax_risk, 'XScale', 'log', ...
             'XTick',      [0.2, 0.5, 1, 2, 5], ...
             'XTickLabel', {'0.2','0.5','1.0','2','5'});
xlim(ax_risk, [0.2, 5]);
ylim(ax_risk, [0, 10]);
xlabel(ax_risk, '\gamma', 'FontSize', 12);
ylabel(ax_risk, 'Out-of-sample test risk',   'FontSize', 12);
%title(ax_risk, 'Ridgeless, $p = 12000$', 'Interpreter', 'latex', 'FontSize', 12);
grid(ax_risk, 'on');
box(ax_risk,  'on');

% ── Legend: one entry per window (top-right) ─────────────────────────────
legend(ax_risk, h_legend, win_labels, ...
    'Interpreter', 'latex', ...
    'Location',    'northeast', ...
    'FontSize',    10, ...
    'Box',         'on');

% ── Save ─────────────────────────────────────────────────────────────────
fname_out = fullfile(out_dir, 'dd_combined.pdf');
exportgraphics(f, fname_out, 'ContentType', 'vector');
close(f);
fprintf('Saved: %s\n', fname_out);