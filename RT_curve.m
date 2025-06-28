function RT_curve(data)

%{
    Plot the RT against the train and test trials.
    
    USAGE: RT_curve()

    Called by: plot_all_figures()
%}

if nargin<1; load('actionChunk_data.mat'); end
sim = data(1).sim;
revis = data(1).revis;
nSubj = length(data);
% if nargin>1
%     if contains(lesion, 'entropyonly')
%         for s = 1:nSubj
%             data(s).rt = data(s).rt_lesion(:,1);
%         end
%     elseif contains(lesion, 'costonly')
%         for s = 1:nSubj
%             data(s).rt = data(s).rt_lesion(:,2);
%         end
%     end
% end
%% Train RT
cmap =  [141 182 205
    255 140 105] / 255;
conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns6,random_train', 'Ns6,structured_train'};
nPresent = sum(data(1).s(strcmp(data(1).cond, conds{1}))==1);
RT = nan(length(conds), nSubj, nPresent);
for s = 1:nSubj
    for condIdx = 1:length(conds)
        states = data(s).s(strcmp(data(s).cond, conds{condIdx}));
        rt = data(s).rt(strcmp(data(s).cond, conds{condIdx}));
        rt_by_nPres = nan(length(unique(states)), nPresent);
        for i = 1:length(unique(states))
            rt_by_nPres(i,:) = rt(states==i);
        end
        RT(condIdx,s,:) = nanmean(rt_by_nPres, 1);
    end
end

avgRT = squeeze(nanmean(RT, 2));
figure; hold on; subplot 121; hold on;
colororder(gca,cmap);
X = 1:nPresent;
for i = 1:2
    plot(X, avgRT(i,:),'-.', 'LineWidth', 3.5);
end
for i = 3:4
    plot(X, avgRT(i,:), 'LineWidth', 3.5);
end

legend({'Ns=4 Random Train', 'Ns=4 Structured Train', 'Ns=6 Random Train',...
    'Ns=6 Structured Train'}, 'Location', 'southeast');
legend('boxoff');
xlabel('Trials'); ylabel('Average RTs (ms)');
title('Train RTs (ms)');

ylim([600 1000])


%% Test RTs
cmap = [238 201 0
    155 205 155] / 255;
conds = {'Ns4,structured_test','Ns4,random_test', 'Ns6,structured_test','Ns6,random_test'};
nPresent = sum(data(1).s(strcmp(data(1).cond, conds{1}))==1);

RT = nan(length(conds), nSubj, nPresent);
for s = 1:nSubj
    for condIdx = 1:length(conds)
        states = data(s).s(strcmp(data(s).cond, conds{condIdx}));
        rt = data(s).rt(strcmp(data(s).cond, conds{condIdx}));
        rt_by_nPres = nan(length(unique(states)), nPresent);
        for i = 1:length(unique(states))
            rt_by_nPres(i,:) = rt(states==i);
        end
        RT(condIdx,s,:) = nanmean(rt_by_nPres, 1);
    end
end

avgRT = squeeze(nanmean(RT, 2));
subplot 122; hold on;
colororder(gca,cmap);
X = 1:nPresent;
for i = 1:2
    plot(X, avgRT(i,:),'-.', 'LineWidth', 3.5);
end
for i = 3:4
    plot(X, avgRT(i,:), 'LineWidth', 3.5);
end

legend({'Ns=4 Structured Test', 'Ns=4 Random Test', 'Ns=6 Structured Test',...
    'Ns=6 Random Test'}, 'Location', 'southeast');
legend('boxoff');
xlabel('Trials'); ylabel('Average RT (ms)');
title('Test RTs (ms)');
ylim([200 1000])

set(gcf, 'Position',  [400, 400, 600, 350])
%exportgraphics(gcf,[pwd '/figures/raw/' sim 'RTTrials.pdf'], 'ContentType', 'vector');
end