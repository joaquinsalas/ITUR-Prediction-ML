% ------------------------------------------------------------
% Count missing values per column and draw histograms with descriptive labels
% ------------------------------------------------------------
clear; clc;

% ---- Input / Output paths
csvFile = '../data/ITUR_resultados_nacional_v1.csv';
figDir  = '../figures';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

% ---- Read data as table
T = readtable(csvFile);

% ---- Descriptive labels
labelMap = containers.Map( ...
    {'TAM_POB', 'DEN_POB', 'DILOCCON50', 'CAR_SER_VI', ...
     'P_USOSUEPV', 'USO_SUECON', 'COND_ACCE', 'EQUIP_URB', 'RESUL_ITUR'}, ...
    {'Tamaño de Población', ...
     'Densidad de Población', ...
     'Distancia a localidades de más de 50k hab.', ...
     'Carencia de servicios de vivienda', ...
     'Proporción de uso de suelo productivo-vegetación', ...
     'Uso de suelo construido', ...
     'Condiciones de accesibilidad', ...
     'Equipamiento urbano', ...
     'ITUR', ...
     'Resultado ITUR'} ...
);

% ---- Apply dependency rule: if TAM_POB is NaN, DEN_POB is NaN
if any(strcmp(T.Properties.VariableNames, 'TAM_POB')) && any(strcmp(T.Properties.VariableNames, 'DEN_POB'))
    nanIdx = isnan(T.TAM_POB);
    T.DEN_POB(nanIdx) = NaN;
end

% ---- Initialize
nCols = width(T);
missingCounts = zeros(1, nCols);

% ---- Process each column
for i = 1:nCols
    colName = T.Properties.VariableNames{i};
    colData = T.(i);
    
    % Count missing values
    if isnumeric(colData)
        missingCounts(i) = sum(isnan(colData));
    else
        missingCounts(i) = sum(ismissing(colData));
    end
    
    % Draw histogram for numeric columns
    if isnumeric(colData)
        figure('Visible','off');
        histTitle = labelMap(colName);
        histogram(colData, 50);
        title([histTitle], 'Interpreter', 'none');
        xlabel(histTitle, 'Interpreter', 'none');
        ylabel('Frecuencia');
        grid on;
        
        % Save figure
        saveas(gcf, fullfile(figDir, [colName, '_hist.png']));
        close(gcf);
    end
end

% ---- Display missing value summary
colNames = T.Properties.VariableNames';
result = table(colNames, missingCounts', 'VariableNames', {'Column', 'MissingRows'});
disp(result);



