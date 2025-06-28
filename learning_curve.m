function learning_curve(data)

%{
    Plot the task performance (accuracy and RT) against the train and test trials.
    
    USAGE: learning_curve()

    Called by: plot_all_figures()
%}


if nargin<1; load('actionChunk_data.mat'); end
sim = data(1).sim;
revis = data(1).revis;
%% Train Accuracy
cmap =  [141 182 205
    255 140 105] / 255;
conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns6,random_train', 'Ns6,structured_train'};

nSubj = length(data);
nPresent = sum(data(1).s(strcmp(data(1).cond, conds{1}))==1);

accuracy = nan(length(conds), nSubj, nPresent);
for s = 1:nSubj
    for c = 1:length(conds)
        states = data(s).s(strcmp(data(s).cond, conds{c}));
        acc = data(s).acc(strcmp(data(s).cond,conds{c}));
        acc_by_nPres = nan(length(unique(states)), nPresent);
        for i = 1:length(unique(states))
            acc_by_nPres(i,:) = acc(states==i);
        end
        accuracy(c,s,:) = nanmean(acc_by_nPres, 1);
    end
end

avgAccuracy = squeeze(nanmean(accuracy, 2));
figure; hold on; subplot 121; hold on;
colororder(gca,cmap);
X = 1:nPresent;
for i = 1:2
    plot(X, avgAccuracy(i,:),'-.', 'LineWidth', 3.5);
end
for i = 3:4
    plot(X, avgAccuracy(i,:), 'LineWidth', 3.5);
end

legend({'Ns=4 Random Train', 'Ns=4 Structured Train', 'Ns=6 Random Train',...
    'Ns=6 Structured Train'}, 'Location', 'southeast');
legend('boxoff');
xlabel('Trials'); ylabel('Average Accuracy');
title('Train Accuracy');

ylim([0 1])

%% Test Accuracy
cmap = [  238 201 0
    155 205 155] / 255;
conds = {'Ns4,structured_test','Ns4,random_test', 'Ns6,structured_test','Ns6,random_test'};

nSubj = length(data); 
nPresent = sum(data(1).s(strcmp(data(1).cond, conds{1}))==1);

accuracy = nan(length(conds), nSubj, nPresent);
for s = 1:nSubj
    for c = 1:length(conds)
        states = data(s).s(strcmp(data(s).cond, conds{c}));
        acc = data(s).acc(strcmp(data(s).cond, conds{c}));
        acc_by_nPres = nan(length(unique(states)), nPresent);
        for i = 1:length(unique(states))
            acc_by_nPres(i,:) = acc(states==i);
        end
        accuracy(c,s,:) = nanmean(acc_by_nPres, 1);
    end
end

avgAccuracy = squeeze(nanmean(accuracy, 2));
subplot 122; hold on;
colororder(gca,cmap);
X = 1:nPresent;
for i = 1:2
    plot(X, avgAccuracy(i,:),'-.', 'LineWidth', 3.5);
end
for i = 3:4
    plot(X, avgAccuracy(i,:), 'LineWidth', 3.5);
end

legend({'Ns=4 Structured Test', 'Ns=4 Random Test', 'Ns=6 Structured Test',...
    'Ns=6 Random Test'}, 'Location', 'southeast');
legend('boxoff');
xlabel('Trials'); ylabel('Average Accuracy');
title('Test Accuracy');
ylim([0 1])

 set(gcf, 'Position',  [400, 400, 600, 350])
 
%exportgraphics(gcf,[pwd '/figures/raw/' sim 'accuracyTrials.pdf'], 'ContentType', 'vector'); 

end