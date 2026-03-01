classdef viewITUR
    properties

    end

    methods


          function     saveFigure(obj, xlab, ylab,zlab, btitle, fn_out, style)
            %black background
            if strcmp(style, 'normal')
                bg = 'w'; fg = 'k';
                bg_vec = [0,0,0]; fg_vec = [1,1,1];
            else
                fg = 'w'; bg = 'k';
                fg_vec = [0,0,0]; bg_vec = [1,1,1];

            end

            %large font size
            set(gca, 'FontSize', 14)

            %labels using the LaTeX interpreter
            xlabel(xlab,  'Interpreter','LaTex','FontSize', 14,'color',fg)
            ylabel(ylab,  'Interpreter','LaTex','FontSize', 14,'color',fg)
            zlabel(zlab,  'Interpreter','LaTex','FontSize', 14,'color',fg)
            title(btitle,  'Interpreter','LaTex','FontSize', 18,'color',fg)

            set(gcf,'Color',fg_vec); % color of the frame around the figure
            set(gca,'Color',bg)%color for the plot area
            set(gca,'XColor',bg_vec); % Set RGB value to what you want
            set(gca,'YColor',bg_vec); % Set RGB value to what you want
            set(gca,'ZColor',bg_vec); % Set RGB value to what you want

            %save the figure
            F = getframe(gcf);
            im = frame2im(F);

            imwrite(im,fn_out);
          end


          function plot_iturs(obj)
% plot_iturs  Plot x_center vs y_center from AlphaEarth_Aguascalientes_ITUR.csv
%
% Usage:
%   plot_iturs()
%
% The function reads '../data/AlphaEarth_Aguascalientes_ITUR.csv' and
% generates a scatter plot of x_center vs y_center.

    % --- File path ---
    csv_file = '../data/AlphaEarth_Aguascalientes_ITUR.csv';
    if ~isfile(csv_file)
        error('File not found: %s', csv_file);
    end

    % --- Read data ---
    data = readtable(csv_file);

    % --- Check required columns ---
    required = {'x_center', 'y_center'};
    if ~all(ismember(required, data.Properties.VariableNames))
        error('The file must contain columns: x_center and y_center.');
    end

    % --- Plot ---
    figure;
    scatter(data.x_center, data.y_center, 2, [0,1,0], 'filled');
    xlabel('x center');
    ylabel('y center');
    title('ITUR locations Aguascalientes');
    axis equal;
    grid on;
     obj.saveFigure('x center','y center','', 'ITUR locations Aguascalientes', '../figures/ITUR_locations_ags.png', 'black')


end



        function view_samples(obj)
            % ---------------------------------------------------------------
            % draw_histogram_RESUL_ITUR.m
            % ---------------------------------------------------------------
            % Reads the CSV file ../data/AlphaEarth_Aguascalientes_ITUR.csv
            % and plots a histogram of the column RESUL_ITUR.
            % ---------------------------------------------------------------

            % --- Parameters ---
            filename = '../data/AlphaEarth_Aguascalientes_ITUR.csv';
            columnName = 'RESUL_ITUR';

            % --- Read CSV file ---
            data = readtable(filename);

            % --- Check if column exists ---
            if ~ismember(columnName, data.Properties.VariableNames)
                error('Column "%s" not found in %s.', columnName, filename);
            end

            % --- Extract data and remove missing values ---
            values = data.(columnName);
            values = values(~ismissing(values));

            % --- Plot histogram ---
            figure(1)
            histogram(values,200, 'Normalization', 'probability','FaceColor',[0,1,0], 'EdgeColor',[0,1,0]);
          
            grid on;
            obj.saveFigure('ITUR', 'Probability','','ITUR Aguascalientes 2020', '../figures/histogram_ITUR_ags.png', 'black')



            figure(2);
            plot(sort(values), '.g');
            hold on
            yline(0.268545,'r', 'LineWidth',3)
            yline(0.464279,'r', 'LineWidth',3)
            yline(0.667117,'r', 'LineWidth',3)
            yline(0.878388,'r', 'LineWidth',3)
            hold off
 
            grid on;

        
           
            obj.saveFigure('item', 'value','', 'ordered arrangement of items by value', '../figures/ordered_ITUR_ags_lines.png', 'black')


            figure(3)
            plot(sort(values),'.g')
            xlabel('item')
            ylabel('value')
            title('ordered arrangement of items by value')

            obj.saveFigure('item', 'value','','ordered arrangement of items by value', '../figures/ordered_ITUR_ags.png', 'black')



        end


        function evaluate_constant_r2(obj)
% evaluate_constant_r2
%   Compute the R² score when predicting a constant (mean) value
%   for RESUL_ITUR >= 0.878389 in AlphaEarth_Aguascalientes_ITUR.csv.

    % --- Parameters ---
    csv_file = '../data/AlphaEarth_Aguascalientes_ITUR.csv';
    tau = 0.878389;

    % --- Load data ---
    if ~isfile(csv_file)
        error('File not found: %s', csv_file);
    end
    data = readtable(csv_file);

    if ~ismember('RESUL_ITUR', data.Properties.VariableNames)
        error('The file must contain the column RESUL_ITUR.');
    end

    % --- Filter subset ---
    subset = data.RESUL_ITUR(data.RESUL_ITUR >= tau);
    if isempty(subset)
        error('No rows satisfy RESUL_ITUR >= %.6f.', tau);
    end

    % --- Constant predictor (mean) ---
    y_true = subset;
    y_pred = mean(y_true) * ones(size(y_true));

    % --- Compute R² ---
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2 = 1 - ss_res / ss_tot;

    % --- Display result ---
    fprintf('Constant predictor (mean = %.6f)\n', mean(y_true));
    fprintf('R² for constant prediction = %.6f\n', r2);
end



    end
end
