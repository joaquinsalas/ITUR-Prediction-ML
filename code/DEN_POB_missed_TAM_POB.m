% ------------------------------------------------------------
% Histogram of DEN_POB where TAM_POB is missing
% ------------------------------------------------------------

 

% Load data
data = readtable('../data/ITUR_resultados_nacional_v1.csv');

% Identify rows with missing TAM_POB
missing_idx = isnan(data.TAM_POB);

tam_pob = data.TAM_POB((~missing_idx) & (data.TAM_POB>0)); 
tam_pob_log = log(tam_pob);

% Extract corresponding DEN_POB values
den_missing = data.DEN_POB(missing_idx);

% Remove any NaNs in DEN_POB
den_missing = den_missing(~isnan(den_missing));

indx = find(data.DILOCCON50<0);
v = data.TAM_POB(indx);
indx_num = find(~isnan(v));
histogram(v(indx_num))
xlabel('tamaño de la poblacion')
ylabel('frequencia')
title('freq de tam pob para dist a loc neg ')



% Plot histogram
figure(1);
histogram(den_missing, 100, 'BinWidth', 0.05, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
xlabel('DEN\_POB');
ylabel('Frequency');
title('Histogram of DEN\_POB where TAM\_POB is missing');
grid on;
