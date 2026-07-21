function Xout   = volstdbwd(X,win)
% This function standardizes X by volatility by expanding/rolling window

T               = size(X,1);
Xout            = nan(size(X));
if isempty(win)
    Xtmp            = X(1:36,:);
    Xout(1:36,:)    = Xtmp./nanstd(Xtmp);
else
    Xtmp            = X(1:win,:);
    Xout(1:win,:)   = Xtmp./nanstd(Xtmp);
end
for t=36+1:T
    if isempty(win)
        % if the win argument is empty, this defaults to using an expanding window
        Xtmp        = X(1:t,:);
    else
        % if win is non-empty, it uses a rolling window of length win
        Xtmp        = X(max(1,t-win+1):t,:);
    end
    Xout(t,:)   = X(t,:)./nanstd(Xtmp);
end