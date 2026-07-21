function [X_boot, Y_out, dates_out] = blocked_bootstrap(X, Y, dates, L, seed)
% BLOCKED_BOOTSTRAP Trims early data, then shuffles X blocks 
% while keeping Y and dates chronological. 
% If seed = 0, the function only trims the data and skips the bootstrap.
%
% Inputs:
%   X      - Predictor matrix [T x d]
%   Y      - Target vector [T x 1]
%   dates  - Date matrix [T x columns]
%   L      - Scalar, block size
%   seed   - Integer for reproducibility (0 to skip shuffling)

    % 1. Set seed (skip if seed is 0)
    if nargin > 4 && ~isempty(seed) && seed ~= 0
        rng(seed);
    end

    T_orig = size(X, 1);
    
    % 2. Trim from the top (earliest dates) to ensure divisibility by L
    num_to_delete = mod(T_orig, L);
    if num_to_delete > 0
        X = X((num_to_delete + 1):end, :);
        Y = Y((num_to_delete + 1):end, :);
        dates = dates((num_to_delete + 1):end, :);
    end
    
    % Keep Y and dates chronological (using the trimmed versions)
    Y_out = Y;
    dates_out = dates;
    
    % 3. Check for seed == 0 condition (Trim only, no bootstrap)
    if nargin > 4 && ~isempty(seed) && seed == 0
        X_boot = X;
        return; % Exit the function early
    end
    
    % --- Proceed with Blocked Bootstrap if seed ~= 0 ---
    [T_new, d] = size(X);
    num_blocks = T_new / L;

    % 4. Moving Block selection for X
    % We allow blocks to start at any index that provides a full length L
    max_start_idx = T_new - L + 1;
    start_indices = randi([1, max_start_idx], 1, num_blocks);
    
    % 5. Reassemble X_boot
    X_boot = zeros(T_new, d);
    for i = 1:num_blocks
        s_idx = start_indices(i);
        target_rows = ((i-1)*L + 1) : (i*L);
        X_boot(target_rows, :) = X(s_idx : s_idx + L - 1, :);
    end
    
end