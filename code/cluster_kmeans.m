classdef cluster_kmeans
    properties
    end

    methods
        function view(obj)
            infile = '../data/accepted_embeddings.csv';
            T = readtable(infile);

            % Paper IDs
            pid = obj.resolve_pid(T);

            % Embeddings
            vnames = string(T.Properties.VariableNames);
            embCols = startsWith(vnames, "e_");
            if ~any(embCols)
                error('No embedding columns found (expected e_0, e_1, ...).');
            end
            X = table2array(T(:, embCols));

            % t-SNE
            rng(42);
            Xz = zscore(X);
            Y = tsne(Xz, 'NumDimensions', 2, ...
                'Perplexity', min(30, max(5, floor(size(Xz,1)/5))), ...
                'Standardize', false, 'Verbose', 1);

            % Plot
            figure(1); clf
            scatter(Y(:,1), Y(:,2), 24, 'filled'); grid on;
            xlabel('t-SNE 1'); ylabel('t-SNE 2');
            title('t-SNE of Accepted Abstract Embeddings');

            showN = min(50, numel(pid));
            idx = round(linspace(1, numel(pid), showN));
            hold on;
            text(Y(idx,1), Y(idx,2), "  " + pid(idx), 'FontSize', 8, 'Interpreter', 'none');
            hold off
            axis equal

            exportgraphics(gcf, '../figures/accepted_embeddings_tsne.png', 'Resolution', 200);
        end

        function place(obj)
            infile   = '../data/accepted_embeddings.csv';
            perWall  = 31;
            numWalls = 4;
            N_expected = perWall * numWalls;

            T = readtable(infile);
            paperID = obj.resolve_pid(T);

            names = string(T.Properties.VariableNames);
            embCols = startsWith(names, "e_");
            if ~any(embCols)
                error('No embedding columns found. Expected e_0, e_1, ...');
            end
            X = table2array(T(:, embCols));
            N = size(X,1);
            if N ~= N_expected
                error('This layout expects %d posters (31/sector). Your file has %d rows.', N_expected, N);
            end

            D = pdist2(X, X, 'cosine');
            D(1:N+1:end) = 0;

            unvisited = true(1, N);
            [~, startNode] = max(mean(D, 2));
            route = zeros(1, N);
            route(1) = startNode;
            unvisited(startNode) = false;

            for k = 2:N
                last = route(k-1);
                drow = D(last, :);
                drow(~unvisited) = inf;
                [~, nextNode] = min(drow);
                route(k) = nextNode;
                unvisited(nextNode) = false;
            end

            improved = true;
            while improved
                improved = false;
                for i = 1:N-2
                    for j = i+2:N
                        if i == 1 && j == N, continue; end
                        a = route(i);   b = route(i+1);
                        c = route(j);   d = route(mod(j, N)+1);
                        oldCost = D(a,b) + D(c,d);
                        newCost = D(a,c) + D(b,d);
                        if newCost + 1e-12 < oldCost
                            route(i+1:j) = route(j:-1:i+1);
                            improved = true;
                        end
                    end
                end
            end

            cx = 0; cy = 0; R = 5;
            coords  = zeros(N,2);
            theta   = zeros(N,1);
            sector  = zeros(N,1);
            idxInSector = zeros(N,1);

            for k = 1:N
                th = -2*pi*(k-1)/N - pi/2;
                coords(k,:) = [cx + R*cos(th), cy + R*sin(th)];
                theta(k) = th;
                sector(k) = floor((k-1)/perWall) + 1;
                idxInSector(k) = k - (sector(k)-1)*perWall;
            end

            orderedIDs = paperID(route);
            out = table( (1:N).', orderedIDs, sector, idxInSector, ...
                rad2deg(mod(theta,2*pi)), coords(:,1), coords(:,2), ...
                'VariableNames', {'PlacementOrder','PaperID','Sector','IndexInSector','ThetaDeg','X','Y'});

            figure('Color','w'); clf; hold on; axis equal; grid on;
            thv = linspace(0, 2*pi, 400);
            plot(cx + R*cos(thv), cy + R*sin(thv), 'k-', 'LineWidth', 1.25);
            scatter(out.X, out.Y, 36, out.PlacementOrder, 'filled');
            colormap turbo; colorbar;
            title('Poster placement on a circle (similar adjacencies)');
            xlabel('X'); ylabel('Y');

            showN = min(40, N);
            idx = round(linspace(1, N, showN));
            text(out.X(idx)+0.1, out.Y(idx), string(out.PaperID(idx)), 'FontSize', 7, 'Interpreter','none');
        end

        function pca_spectrum(obj)
            infile = '../data/accepted_embeddings.csv';
            T = readtable(infile);
            vnames = string(T.Properties.VariableNames);
            embCols = startsWith(vnames, "e_");
            if ~any(embCols)
                error('No embedding columns found (expected e_*).');
            end
            X = table2array(T(:, embCols));
            Xc = X - mean(X,1);
            [~, S, ~] = svd(Xc, 'econ');
            sigma = diag(S);
            sigma_norm = sigma / sum(sigma);
            csum = cumsum(sigma_norm);

            figure('Color','w'); clf;
            plot(1:numel(csum), csum, '-o', 'MarkerSize', 3, 'LineWidth', 1.2);
            grid on; box on;
            xlabel('Rank r'); ylabel('Cumulative sum of normalized singular values');
            title('PCA spectrum (cum. normalized singular values)'); ylim([0, 1.02]);
            hold on;
            thrs = [0.80 0.90 0.95 0.99];
            colors = lines(numel(thrs));
            for i = 1:numel(thrs)
                yline(thrs(i), '--', sprintf('%.0f%%', thrs(i)*100), ...
                    'LabelHorizontalAlignment','left', 'Color', colors(i,:), 'Alpha', 0.7);
            end
            hold off;

            fprintf('\nCUMULATIVE SUM OF NORMALIZED SINGULAR VALUES (sigma/sum(sigma)):\n');
            fprintf('r\tcum_sum\n');
            for r = 1:numel(csum)
                fprintf('%d\t%.6f\n', r, csum(r));
            end
            fprintf('\nCutoff ranks (smallest r with cum_sum >= threshold):\n');
            for i = 1:numel(thrs)
                rstar = find(csum >= thrs(i), 1, 'first');
                if isempty(rstar)
                    fprintf('  %.0f%%:\tNot reached\n', thrs(i)*100);
                else
                    fprintf('  %.0f%%:\tr = %d (cum_sum = %.6f)\n', thrs(i)*100, rstar, csum(rstar));
                end
            end
            fprintf('\n');
        end

        function cluster(obj, ks)
            % Runs k-means for k ∈ ks (default 2:10),
            % outputs one CSV per k with columns:
            % Cluster, PaperID, Title
            % Ordered by Cluster (group), then PaperID.

            if nargin < 2 || isempty(ks)
                ks = 2:10;
            end
            infile_embed = '../data/accepted_embeddings.csv';
            infile_meta  = '../data/accepted_submissions.csv';
            outdir = '../data/clusterings';
            if ~exist(outdir, 'dir'), mkdir(outdir); end

            % --- Load embeddings
            Te = readtable(infile_embed);
            paperID = obj.resolve_pid(Te);

            vnames = string(Te.Properties.VariableNames);
            embCols = startsWith(vnames, "e_");
            if ~any(embCols)
                error('No embedding columns found. Expected e_0, e_1, ...');
            end
            X = table2array(Te(:, embCols));
            Xz = zscore(X);
            rng(42);

            % --- Load titles from submissions
            Ts = readtable(infile_meta);
            pid_sub = obj.resolve_pid(Ts);
            titleVar = obj.resolve_title(Ts);
            Tmeta = table(pid_sub, titleVar, 'VariableNames', {'PaperID','Title'});

            % --- Loop over k
            for k = ks
                fprintf('Running k-means with k = %d...\n', k);
                [idx, ~] = kmeans(Xz, k, ...
                    'Distance','sqeuclidean', ...
                    'Replicates', 10, ...
                    'MaxIter', 1000, ...
                    'Display', 'off');

                % Build and merge
                Tout = table(paperID, idx, 'VariableNames', {'PaperID','Cluster'});
                Tmerge = outerjoin(Tout, Tmeta, 'Keys','PaperID', ...
                    'MergeKeys', true, 'Type','left');

                % Order columns and sort by group
                Tmerge = movevars(Tmerge, {'Cluster','PaperID','Title'}, 'Before', 1);
                Tmerge = Tmerge(:, {'Cluster','PaperID','Title'});

                % Sort by Cluster then PaperID
                Tmerge = sortrows(Tmerge, {'Cluster','PaperID'});

                % Save CSV
                outf = fullfile(outdir, sprintf('kmeans_k%d.csv', k));
                writetable(Tmerge, outf);
                fprintf('  -> wrote %s\n', outf);
            end

            fprintf('Done.\n');
        end
    end

    methods (Access = private)
        function pid = resolve_pid(~, T)
            % Robustly extract PaperID as string from a table with one of:
            % 'PaperID', 'Paper_ID', 'Paper ID'
            v = string(T.Properties.VariableNames);
            if any(strcmpi(v, 'PaperID'))
                pid = string(T.PaperID);
            elseif any(strcmpi(v, 'Paper_ID'))
                pid = string(T.Paper_ID);
            elseif any(strcmpi(v, 'Paper ID'))
                pid = string(T.("Paper ID"));
            else
                error('Could not find a Paper ID column.');
            end
        end

        function titleVar = resolve_title(~, T)
            % Robustly extract Title column (case-insensitive)
            v = string(T.Properties.VariableNames);
            hit = find(strcmpi(v, 'Title'), 1);
            if isempty(hit)
                error('Could not find a Title column in ../data/accepted_submissions.csv');
            end
            titleVar = string(T.(v(hit)));
        end
    end
end
