

data = readtable('../data/itur_difference_regression_dataset.csv');
% Display the first few rows of the data
%head(data);

y = data.diff(:);
thr = 0.01;
indx = find(abs(y) > thr);

%% -------------------------------------------------
%  t-SNE on hat_ variables
%% -------------------------------------------------

% 1) Extract hat_ variables
vars = data.Properties.VariableNames;
hat_mask = startsWith(vars,'hat_');
hat_vars = vars(hat_mask);

X = data{:,hat_vars};   % feature matrix

% 2) Define splits
all_idx = (1:size(X,1))';
non_idx = setdiff(all_idx, indx);

% 3) Sample 10,000 records from non_idx
rng(42)
n_sample = min(10000,length(non_idx));
sample_non = randsample(non_idx, n_sample);

X_train = X(sample_non,:);
X_test  = X(indx,:);

% 4) Standardize using training statistics
mu = mean(X_train);
sigma = std(X_train);
sigma(sigma==0) = 1;

X_train = (X_train - mu)./sigma;
X_test  = (X_test  - mu)./sigma;

% 5) Fit t-SNE on training subset
Y_train = tsne(X_train, ...
    'NumDimensions',2, ...
    'Perplexity',30, ...
    'Standardize',false);

%% 6) Learn joint mapping X -> R^2

% One network with 2 outputs
mdl = fitrnet( ...
    X_train, Y_train, ...
    'LayerSizes',[64 32], ...
    'Activations','relu', ...
    'Standardize',false);

% Predict both coordinates simultaneously
Y_test = predict(mdl, X_test);

%% 7) Plot with translucency and size ~ |diff|

% Magnitude of diff
mag_train = abs(y(sample_non));
mag_test  = abs(y(indx));

% Normalize sizes (avoid extreme scaling)
all_mag = [mag_train; mag_test];
min_m = min(all_mag);
max_m = max(all_mag);

% Size range
s_min = 10;
s_max = 120;

size_train = s_min + (mag_train - min_m)/(max_m - min_m) * (s_max - s_min);
size_test  = s_min + (mag_test  - min_m)/(max_m - min_m) * (s_max - s_min);

figure(1)
clf

h1 = scatter(Y_train(:,1),Y_train(:,2), ...
    size_train,'b','filled');
hold on
h2 = scatter(Y_test(:,1),Y_test(:,2), ...
    size_test,'r','filled');

% Translucency
h1.MarkerFaceAlpha = 0.25;
h2.MarkerFaceAlpha = 0.5;

hold off
axis equal
legend('abs(y) <= thr','abs(y) > thr')
title('t-SNE projection of hat_ variables')
xlabel('Dim 1')
ylabel('Dim 2')
grid on