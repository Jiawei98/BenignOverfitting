%% alpha2_tstat_bar.m
% Bar chart of alpha_2 t-statistics for RFF model.
% One figure per window size: T=12, T=60, T=120.
% 3 bars per z value, one per sub-period.
% Right side: CV result (DemeanX=1, DemeanY=1 only).
% Sub-periods: Pre-1975, Post-1975, Full Sample.

% ── Load data ─────────────────────────────────────────────────────────────
T  = readtable('tables/fix/Alpha2_tstat.xlsx');
CV = readtable('tables/cv/RFF_CV_Results.xlsx');

% ── Settings ──────────────────────────────────────────────────────────────
wins  = [12, 60, 120];
demX  = 1;
demY  = 1;
model = 'RFF';

periods       = {[1930, 1974], [1975, 2020], [1930, 2020]};
period_labels = {'Pre-1975', 'Post-1975', 'Full Sample'};
nP            = numel(periods);

% z columns and x-axis labels
z_vars   = {'z_0',  'z_1', 'z_10', 'z_100', 'z_1000', 'z_10000', 'z_Inf'};
z_labels = {'Ridgeless', '0', '1', '2', '3', '4', '9'};
nZ       = numel(z_vars);

% ── Colors: one per period (purple / teal / coral) ────────────────────────
period_colors = [
    0.49, 0.18, 0.56;   % purple  – Pre-1975
    0.17, 0.63, 0.60;   % teal    – Post-1975
    0.93, 0.40, 0.20;   % coral   – Full Sample
];

out_dir = fullfile('plots');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ── Loop over window sizes ─────────────────────────────────────────────────
for wi = 1:numel(wins)
    win = wins(wi);

    % ── Collect z data matrix [nZ x nP] ───────────────────────────────────
    data_mat = nan(nZ, nP);
    for p = 1:nP
        mask = strcmp(T.Model, model)        & ...
               T.Window     == win           & ...
               T.DemeanX    == demX          & ...
               T.DemeanY    == demY          & ...
               T.Period_beg == periods{p}(1) & ...
               T.Period_end == periods{p}(2);
        row = T(mask, :);
        if ~isempty(row)
            for zi = 1:nZ
                data_mat(zi, p) = row.(z_vars{zi});
            end
        end
    end

    % ── Collect CV values [1 x nP] ────────────────────────────────────────
    cv_vals = nan(1, nP);
    for p = 1:nP
        mask_cv = CV.Window     == win           & ...
                  CV.DemeanX    == demX          & ...
                  CV.DemeanY    == demY          & ...
                  CV.Period_beg == periods{p}(1) & ...
                  CV.Period_end == periods{p}(2);
        row_cv = CV(mask_cv, :);
        if ~isempty(row_cv)
            cv_vals(p) = row_cv.Alpha2_tstat;
        end
    end

    % ── Build combined x positions ────────────────────────────────────────
    x_z  = 1:nZ;       % positions 1..7
    x_cv = nZ + 1;   % close gap after z bars

    % ── Y limits ──────────────────────────────────────────────────────────
    all_vals = [data_mat(:); cv_vals(:)];
    y_pad = 0.3;
    y_lo  = floor(min(all_vals) * 10) / 10 - y_pad;
    y_hi  = ceil( max(all_vals) * 10) / 10 + y_pad;

    % ── Figure ────────────────────────────────────────────────────────────
    f  = figure('Color', 'w', 'Position', [100, 100, 1000, 400], 'Visible', 'off');
    ax = axes('Parent', f);
    hold(ax, 'on');

    % z bars
    b = bar(ax, x_z, data_mat, 'grouped');
    for p = 1:nP
        b(p).FaceColor = period_colors(p, :);
        b(p).EdgeColor = 'none';
    end

    % CV bars
    b_cv = bar(ax, x_cv, cv_vals, 'grouped');
    for p = 1:nP
        b_cv(p).FaceColor = period_colors(p, :);
        b_cv(p).EdgeColor = 'none';
        b_cv(p).HandleVisibility = 'off';
    end

    % Reference lines
    yline(ax, 0,    '-',  'Color', [0.35 0.35 0.35], 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
    yline(ax, 1.96, '--', 'Color', [0.75 0.00 0.00], 'LineWidth', 1.1, ...
        'HandleVisibility', 'off');

    % Vertical separator before CV column
    xline(ax, nZ + 1, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.9, ...
        'HandleVisibility', 'off');

    % Axes formatting
    xlim(ax, [0.5, x_cv + 0.5]);
    ylim(ax, [y_lo, y_hi]);
    set(ax, 'XTick',      [x_z, x_cv], ...
            'XTickLabel', [z_labels, {'CV'}], ...
            'TickLabelInterpreter', 'latex', ...
            'FontSize', 13);
    xlabel(ax, '$z$',            'Interpreter', 'latex', 'FontSize', 14);
    ylabel(ax, '$t_{\alpha_2}$', 'Interpreter', 'latex', 'FontSize', 14);
    box(ax, 'on');
    grid(ax, 'on');
    set(ax, 'GridAlpha', 0.3);

    % Legend
    legend(ax, b, period_labels, ...
        'Location', 'northeast', ...
        'FontSize',  12, ...
        'Box',       'on');

    % ── Save ──────────────────────────────────────────────────────────────
    fname = fullfile(out_dir, sprintf('alpha2_tstat_rff_win%d.png', win));
    saveas(f, fname);
    close(f);
    fprintf('Saved: %s\n', fname);
end