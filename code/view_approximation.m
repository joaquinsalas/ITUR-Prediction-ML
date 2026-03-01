data = readtable('../data/close_estimates.csv');
% Display the first few rows of the data
head(data);

y = data.RESUL_ITUR(:);
yhat = data.ITUR_calc(:);

% Remove NaNs
mask = ~isnan(y) & ~isnan(yhat);
y = y(mask);
yhat = yhat(mask);

SS_res = sum((y - yhat).^2);
SS_tot = sum((y - mean(y)).^2);

R2 = 1 - SS_res / SS_tot;

fprintf('R^2 = %.10f\n', R2);

