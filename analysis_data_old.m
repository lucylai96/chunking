function analysis = analysis_data(plotCase, data)

%{
    Analyze and plot the average accuracy, average RT, intrachunk RT in
    different blocks of the set size manipulation experiment.

    USAGE:
        analysis_data('avgAcc', data)
        analysis_data('avgAcc_train', data)*
        analysis_data('avgRT', data)
        analysis_data('avgRT_train', data)*
        analysis_data('errors_train', data)*

        analysis_data('ICRT_all_train', data)*
        analysis_data('ICRT_all_test', data)
        analysis_data('ICRT_correct_chunk', data)*
        analysis_data('corr_RT_train_test', data)*

        analysis_data('ICRT_trials_Ns4', data)*
        analysis_data('ICRT_trials_Ns6', data)*

        analysis_data('ICRT', data)
        analysis_data('actionSlips', data)
        
        *output stats

    Input: 'plotCase' is a string representing which plot to show; data is
    an optional input and can be the simulated data structure


    Called by: plot_all_figures()
%}

if nargin<2; load('actionChunk_data.mat'); end
%sim = data(1).sim;
%revis = data(1).revis;

%addpath('/Volumes/GoogleDrive/My Drive/Harvard/Projects/mat-tools');

nSubj = length(data);

conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns4,structured_test','Ns4,random_test',...
    'Ns6,random_train', 'Ns6,structured_train', 'Ns6,structured_test','Ns6,random_test'};

cmap =[141 182 205
    255 140 105
    238 201 0
    155 205 155] / 255;

nTrials = 120;
% ICRT preprocessing
ICRT_all = zeros(nSubj, length(conds));
ICRT_correct = zeros(nSubj, length(conds));
ICRT_trial = nan(nSubj, nTrials, length(conds));
NCRT_trial = nan(nSubj, nTrials, length(conds));
chunkInit = [2,5]; % chunk-initiating state is 2 for Ns = 4 and 5 for Ns = 6
for s = 1:nSubj
    for c = 1:length(conds)
        idx = strcmp(data(s).cond, conds(c));
        state = data(s).s(idx);
        action = data(s).a(idx);
        rt = data(s).rt(idx);
        RT(s,c) = nanmean(rt);
        if contains(conds(c),'4')
            condIdx = 1;
        elseif contains(conds(c), '6')
            condIdx = 2;
        end
        ics = find(state==chunkInit(condIdx))+1; ics(ics>length(state))=[];
        ICRT_trial(s,1:length(rt(ics)),c) = rt(ics);
        NCRT_trial(s,1:length(rt(setdiff(1:end,ics))),c) = rt(setdiff(1:end,ics));
        ICRT_all(s,c) = nanmean(rt(ics));
        NCRT_all(s,c) = nanmean(rt(setdiff(1:end,ics)));
        temp_rt = rt; temp_rt(ics) = NaN;
        NCRT_correct(s,c) = nanmean(temp_rt(state==action));
        correct_ics = intersect(find(state == action), ics);
        incorrect_ics = intersect(find(state ~= action), ics);
        %if sim; cost = abs(data(s).cost(idx));
        %    COST(s,c) = nanmean(cost);
        %    cost_correct(s,c) = nanmean(cost(correct_ics));
        %end
        ICRT_correct(s,c) = nanmean(rt(correct_ics)); % correct ICRTs
        ICRT_incorrect(s,c) = nanmean(rt(incorrect_ics)); % incorrect ICRTs
    end
end

%ICRT_correct(isnan(ICRT_correct)) = 0;
%ICRT_incorrect(isnan(ICRT_incorrect)) = 0;
% plotcase
switch plotCase
     case 'ICRT_trials'
        cmap =[238 123 100 % Ns4 Random
            118 181 197
            216 38 0 % Ns6 Random
            30 129 176] / 255;

        nTrials = 20;
        ICRT_all_trials = nan(nSubj, nTrials, length(conds));
        NCRT_all_trials = nan(nSubj, 101, length(conds));
        acc_ICRT = nan(nSubj, nTrials, length(conds));
        acc_NCRT = nan(nSubj, 101, length(conds));
        chunkInit = [2,5]; % chunk-initiating state is 2 for Ns = 4 and 5 for Ns = 6
        for s = 1:nSubj
            for c = 1:length(conds)
                idx = strcmp(data(s).cond, conds(c));
                state = data(s).s(idx);
                action = data(s).a(idx);
                reward = data(s).r(idx);
                rt = data(s).rt(idx);
                if contains(conds(c),'4')
                    condIdx = 1;
                elseif contains(conds(c), '6')
                    condIdx = 2;
                end
                ics = find(state==chunkInit(condIdx))+1; ics(ics>length(state))=[]; % intra-chunk state

                ICRT_all_trials(s,1:length(ics),c) = rt(ics);             % 8 'pages', one for each condition
                NCRT_all_trials(s,1:length(rt(setdiff(1:end,ics))),c) = rt(setdiff(1:end,ics));
                acc_ICRT(s,1:length(ics),c) = reward(ics);          % 8 'pages', one for each condition
                acc_NCRT(s,1:length(rt(setdiff(1:end,ics))),c) = reward(setdiff(1:end,ics));         % 8 'pages', one for each condition
                err_ICRT(s,1:length(ics),c) = reward(ics)==0;          % 8 'pages', one for each condition
                
                %accuracy_ics(s,1:length(ics),c) = reward(ics);
            end
        end

        figure; hold on;  win = 10; colormap(cmap)
        nexttile; hold on; hold on; ylim([450 1020])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,1)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,2)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Intra-chunk RT (ms)'); title('Ns=4')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'}); legend('boxoff')

        nexttile; hold on;  ylim([450 1020])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,5)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,5),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,6)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,6),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('Intra-chunk RT (ms)'); title('Ns=6')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'}); legend('boxoff')

        nexttile; hold on; ylim([0 350])
        ICRT_end = [squeeze(nanmean(nanmean(ICRT_all_trials(:,15:20,1:2),2))) squeeze(nanmean(nanmean(ICRT_all_trials(:,15:20,5:6),2)))];
        ICRT_diff = [squeeze(nanmean(ICRT_all_trials(:,15:20,1),2)-nanmean(ICRT_all_trials(:,15:20,2),2)) squeeze(nanmean(ICRT_all_trials(:,15:20,5),2)-nanmean(ICRT_all_trials(:,15:20,6),2))];
        b = barwitherr([sem(ICRT_diff,1)],[nanmean(ICRT_diff)]);
        b(1).FaceColor = [1 1 1]; b(1).LineWidth = 3; b(2).FaceColor = 'k';
        xticks([b(1).XEndPoints b(2).XEndPoints]); xticklabels({'Ns=4','Ns=6'});
        ylabel('\Delta Intra-chunk RT (ms)');box off;

        win = 30;
        nexttile; hold on; hold on; ylim([450 1020])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,1)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,2)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Non-chunk RT (ms)');

        nexttile; hold on; ylim([450 1020])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,5)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,5),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,6)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,6),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('Non-chunk RT (ms)');

        nexttile; hold on; ylim([0 350])
        NCRT_end = [squeeze(nanmean(nanmean(NCRT_all_trials(:,50:60,1:2),2))) squeeze(nanmean(nanmean(NCRT_all_trials(:,end-10:end,5:6),2)))];
        NCRT_diff = [squeeze(nanmean(NCRT_all_trials(:,50:60:end,1),2)-nanmean(NCRT_all_trials(:,50:60:end,2),2)) squeeze(nanmean(NCRT_all_trials(:,end-10:end,5),2)-nanmean(NCRT_all_trials(:,end-10:end,6),2))];
        b = barwitherr([sem(NCRT_diff,1)],[nanmean(NCRT_diff)]);
        b(1).FaceColor = [1 1 1]; b(1).LineWidth = 3; b(2).FaceColor = 'k';
        xticks([b(1).XEndPoints b(2).XEndPoints]); xticklabels({'Ns=4','Ns=6'});
        ylabel('\Delta Non-chunk RT (ms)');box off;

        set(gcf, 'Position',  [400, 100, 920, 425])


%         figure; hold on;  win = 10; colormap(cmap)
%         nexttile; hold on; hold on; ylim([0 1])
%         h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,1)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
%         h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,2)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
%         xlabel('Trial'); ylabel('Intra-chunk accuracy'); title('Ns=4')
%         legend([h(1).mainLine h(2).mainLine],{'Random','Structured'}); legend('boxoff')
% 
%         nexttile; hold on;  ylim([0 1])
%         h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,5)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,5),1),win,'omitnan'),{'color',cmap(3,:)},1);
%         h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,6)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,6),1),win,'omitnan'),{'color',cmap(4,:)},1);
%         xlabel('Trial');  ylabel('Intra-chunk accuracy'); title('Ns=6')
%         legend([h(1).mainLine h(2).mainLine],{'Random','Structured'}); legend('boxoff')
% 
%              win = 30;
%         nexttile; hold on; hold on; ylim([0 1])
%         h(1) = shadedErrorBar(1:101, movmean(nanmean(acc_NCRT(:,:,1)),win,'omitnan'), movmean(sem(acc_NCRT(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
%         h(2) = shadedErrorBar(1:101, movmean(nanmean(acc_NCRT(:,:,2)),win,'omitnan'), movmean(sem(acc_NCRT(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
%         xlabel('Trial'); ylabel('Non-chunk accuracy');
% 
%         nexttile; hold on; ylim([0 1])
%         h(1) = shadedErrorBar(1:101, movmean(nanmean(acc_NCRT(:,:,5)),win,'omitnan'), movmean(sem(acc_NCRT(:,:,5),1),win,'omitnan'),{'color',cmap(3,:)},1);
%         h(2) = shadedErrorBar(1:101, movmean(nanmean(acc_NCRT(:,:,6)),win,'omitnan'), movmean(sem(acc_NCRT(:,:,6),1),win,'omitnan'),{'color',cmap(4,:)},1);
%         xlabel('Trial'); ylabel('Non-chunk accuracy');


    case 'err_trials'

        
         figure; hold on;  win = 10; colormap(cmap)
        nexttile; hold on; hold on; ylim([0 1])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,1)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,2)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Intra-chunk error'); title('Ns=4')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'},'Location','NorthEast'); legend('boxoff')

        nexttile; hold on;  ylim([0 1])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,5)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,5),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,6)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,6),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial');  ylabel('Intra-chunk error'); title('Ns=6')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'},'Location','NorthEast'); legend('boxoff')

         win = 30;
        nexttile; hold on; hold on; ylim([0 1])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,1)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,2)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Non-chunk error');

        nexttile; hold on; ylim([0 1])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,5)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,5),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,6)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,6),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('Non-chunk error');

        nexttile; hold on;
        % chunks help reduce error more in Ns=6 (greater reduction of error
        % in Ns=6)

        err_mean = [nanmean(1-acc_NCRT(:,50:60,1),2) nanmean(1-acc_NCRT(:,50:60,2),2) nanmean(1-acc_NCRT(:,end-10:end,5),2) nanmean(1-acc_NCRT(:,end-10:end,6),2)];
        err_diff = [err_mean(:,1)-err_mean(:,2) err_mean(:,3)-err_mean(:,4)];
        b = barwitherr([sem(err_diff,1)],[nanmean(err_diff)]);
       %b = barwitherr([sem(err_mean(:,1:2),1);sem(err_mean(:,3:4),1)],[nanmean(err_mean(:,1:2));nanmean(err_mean(:,3:4))]);
        b(1).FaceColor = [1 1 1]; b(1).LineWidth = 3; b(2).FaceColor = 'k';


    case 'avgAcc'
        acc = nan(nSubj, length(conds));
        se = nan(nSubj, length(conds));
        for s = 1:nSubj
            for c = 1:length(conds)
                acc(s,c) = nanmean(data(s).acc(strcmp(data(s).cond, conds{c})));
            end
        end

        figure; hold on;
        colororder(cmap);
        X = 1:2;
        tmp = mean(acc,1); plotAcc(1,:) = tmp(1:length(conds)/2); plotAcc(2,:) = tmp(length(conds)/2+1:length(conds));
        b = bar(X, plotAcc, 'CData', cmap);
        se = nanstd(acc, 1)/sqrt(nSubj) ; se = transpose(reshape(se, [4 2]));
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotAcc, min(se,1-plotAcc), se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        ylim([0 1]);
        lgd = legend('Random Train', 'Structured Train', 'Structured Test', 'Random Test', 'Location', 'north');
        legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('Average accuracy');

        set(gcf, 'Position',  [400, 400, 650, 300])
        analysis.acc = acc;

        %exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

    case 'avgAcc_train'

        acc = nan(nSubj, length(conds));
        se = nan(nSubj, length(conds));
        for s = 1:nSubj
            for c = 1:length(conds)
                acc(s,c) = nanmean(data(s).acc(strcmp(data(s).cond, conds{c})));
            end
        end

        figure; hold on;
        colororder(cmap);
        X = 1:2;
        tmp = mean(acc,1); plotAcc(1,:) = tmp(1:length(conds)/2); plotAcc(2,:) = tmp(length(conds)/2+1:length(conds));
        b = bar(X, plotAcc(:,1:2), 'CData', cmap);
        se = nanstd(acc, 1)/sqrt(nSubj) ; se = transpose(reshape(se, [4 2]));
        errorbar_pos = errorbarPosition(b, se(:,1:2));
        errorbar(errorbar_pos', plotAcc(:,1:2), min(se(:,1:2),1-plotAcc(:,1:2)), se(:,1:2), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        ylim([0 1]);
        lgd = legend('Random Train', 'Structured Train', 'Location', 'north');
        legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('Average accuracy');

        set(gcf, 'Position',  [400, 400, 400, 270])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

        %[p,tbl,stats] = anova1(acc);
        [h,p,ci,stats] = ttest(acc(:,1),acc(:,2));
        disp('Average Accuracy (Random v. Structured Train)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(acc(:,5),acc(:,6));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

    case 'avgRT'
        rt = nan(nSubj, length(conds));
        for s = 1:nSubj
            for c = 1:length(conds)
                rt(s,c) = nanmean(data(s).rt(strcmp(data(s).cond, conds(c))));
                se(s,c) = nanstd(data(s).rt(strcmp(data(s).cond, conds(c))));
                %rt_all{s,c} = rtData(strcmp(data(s).cond, conds(c)));
            end
        end

        figure;
        colororder(cmap);
        hold on;
        tmp = mean(rt,1); plotRT(1,:) = tmp(1:length(conds)/2); plotRT(2,:) = tmp(length(conds)/2+1:length(conds));
        b = bar([1 2], plotRT);
        se = nanstd(rt, 1)/sqrt(nSubj); se = transpose(reshape(se, [length(conds)/2 2]));
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Structured Test', 'Random Test', 'Location', 'north');
        legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('Average RT (ms)'); ylim([0 1000])


        set(gcf, 'Position',  [400, 400, 650, 300])

        %exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');


    case 'avgRT_train'
        conds = {'Ns4,random_train', 'Ns4,structured_train','Ns6,random_train', 'Ns6,structured_train', };
        rt = nan(nSubj, length(conds));
        for s = 1:nSubj
            for c = 1:length(conds)
                rt(s,c) = nanmean(data(s).rt(strcmp(data(s).cond, conds(c))));
                se(s,c) = nanstd(data(s).rt(strcmp(data(s).cond, conds(c))));
            end
        end

        figure;
        colororder(cmap); hold on;
        tmp = mean(rt,1); plotRT(1,:) = tmp(1:length(conds)/2); plotRT(2,:) = tmp(length(conds)/2+1:length(conds));
        b = bar([1 2], plotRT);
        se = nanstd(rt, 1)/sqrt(nSubj); se = transpose(reshape(se, [length(conds)/2 2]));
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Location', 'north');
        legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('Average RT (ms)'); ylim([500 1000])


        set(gcf, 'Position',  [400, 400, 400, 270])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');


        [h,p,ci,stats] = ttest(rt(:,1),rt(:,2));
        disp('Average RT (Random v. Structured Train)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(rt(:,3),rt(:,4));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

    case 'errors_train'
        conds = {'Ns4,random_train', 'Ns4,structured_train','Ns6,random_train', 'Ns6,structured_train', };
        err = nan(nSubj, length(conds));
        for s = 1:nSubj
            for c = 1:length(conds)
                miss = data(s).s ~= data(s).a;
                err(s,c) = nanmean(miss(strcmp(data(s).cond, conds(c))));
                se(s,c) = nanstd(miss(strcmp(data(s).cond, conds(c))));
            end
        end

        figure;
        colororder(cmap); hold on;
        tmp = mean(err,1); plotErr(1,:) = tmp(1:length(conds)/2); plotErr(2,:) = tmp(length(conds)/2+1:length(conds));
        b = bar([1 2], plotErr);
        se = nanstd(err, 1)/sqrt(nSubj); se = transpose(reshape(se, [length(conds)/2 2]));
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotErr, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Location', 'north');
        legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('% Errors'); ylim([0 0.5])

        set(gcf, 'Position',  [400, 400, 400, 270])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');


        [h,p,ci,stats] = ttest(err(:,1),err(:,2));
        disp('Errors (Random v. Structured Train)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(err(:,3),err(:,4));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])


    case 'ICRT'
        figure; hold on; all = 0;
        colororder(cmap);
        tmp = nanmean(ICRT_correct,1); plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = nanstd(ICRT_correct, 1) / sqrt(nSubj); se = transpose(reshape(se, [4 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Structured Test', 'Random Test','Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('Intrachunk Response Time (ms)');
        if all
            title('ICRTs - All Trials')
        else
            title('ICRTs - Correct Trials')
        end

        set(gcf, 'Position',  [400, 400, 650, 300])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

    case 'ICRT_all_train'
        figure; hold on;
        colororder(cmap(1:2,:));
        tmp = nanmean(ICRT_all, 1); plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = nanstd(ICRT_all, 1) / sqrt(nSubj); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT(:,1:2));
        errorbar_pos = errorbarPosition(b, se(:,1:2));
        errorbar(errorbar_pos', plotICRT(:,1:2), se(:,1:2), se(:,1:2), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        ylim([200 1000])

        set(gcf, 'Position',  [400, 400, 400, 270])

        % More reduction in chunking?
        [h,p,ci,stats] = ttest(ICRT_all(:,1)-ICRT_all(:,2),ICRT_all(:,5)-ICRT_all(:,6));
        disp('More reduction in chunking in Ns = 6 than Ns = 4? (All trials)')
        disp(strcat('Reject null?:',num2str(h))); disp(strcat('p-value:',num2str(p)));

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

        [h,p,ci,stats] = ttest(ICRT_all(:,1),ICRT_all(:,2));
        disp('Average ICRT - All (Random v. Structured Train)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(ICRT_all(:,5),ICRT_all(:,6));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

    case 'ICRT_all_test'
        figure; hold on;
        tmp = nanmean(ICRT_all, 1); plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = nanstd(ICRT_all, 1) / sqrt(nSubj); se = transpose(reshape(se, [length(conds)/2 2]));
        colororder(cmap(3:4,:));
        b = bar([1 2], plotICRT(:,3:4));
        errorbar_pos = errorbarPosition(b, se(:,3:4));
        errorbar(errorbar_pos', plotICRT(:,3:4), se(:,3:4), se(:,3:4), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Structured Test', 'Random Test', 'Location', 'northwest');  legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('Intrachunk Response Time (ms)');

        set(gcf, 'Position',  [400, 400, 400, 270])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');



    case 'ICRT_correct_chunk'
        conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns6,random_train', 'Ns6,structured_train'};

        % split into people who reduced ICRT from random to structured vs
        % those who did not reduce ICRT from random to structured

        chunking(:,1) = ICRT_all(:,1)-ICRT_all(:,2)>0;              % Ns4: chunk n = 57 | noChunk n = 19
        chunking(:,2) = ICRT_all(:,5)-ICRT_all(:,6)>0;              % Ns6: chunk n = 53 | noChunk n = 23
        %chunking = logical(ones(nSubj,2));

        %save('chunking.mat','chunking')
        %load 'chunking_data.mat'

        ICRT_correct_chunk_Ns4 = ICRT_correct(chunking(:,1),1:4);   % get the corresponding test RTs
        ICRT_correct_noChunk_Ns4 = ICRT_correct(~chunking(:,1),1:4);
        ICRT_correct_chunk_Ns6 = ICRT_correct(chunking(:,2),5:8);   % get the corresponding test RTs
        ICRT_correct_noChunk_Ns6 = ICRT_correct(~chunking(:,2),5:8);

        ICRT_chunk_Ns4 = ICRT_all(chunking(:,1),1:4);
        ICRT_noChunk_Ns4 = ICRT_all(~chunking(:,1),1:4);
        ICRT_chunk_Ns6 = ICRT_all(chunking(:,2),5:8);
        ICRT_noChunk_Ns6 = ICRT_all(~chunking(:,2),5:8);

        RT_chunk_Ns4 = RT(chunking(:,1),1:4);
        RT_noChunk_Ns4 = RT(~chunking(:,1),1:4);
        RT_chunk_Ns6 = RT(chunking(:,2),5:8);
        RT_noChunk_Ns6 = RT(~chunking(:,2),5:8);

        % the change in policy complexity predicts the change in ICRT
        % for real data
        %load complexity.mat
        % maybe calculate complexity only on correct trials?
        %figure; hold on;
        %plot([complexity(chunking(:,1),1)-complexity(chunking(:,1),2)],[ICRT_correct_chunk_Ns4(:,1)-ICRT_correct_chunk_Ns4(:,2)],'.','MarkerSize',20); lsline
        %plot([complexity(chunking(:,1),3)-complexity(chunking(:,1),4)],[ICRT_correct_chunk_Ns4(:,3)-ICRT_correct_chunk_Ns4(:,4)],'.','MarkerSize',20); lsline
        %plot([complexity(chunking(:,2),5)-complexity(chunking(:,2),6)],[ICRT_correct_chunk_Ns6(:,1)-ICRT_correct_chunk_Ns6(:,2)],'.','MarkerSize',20); lsline
        %plot([complexity(chunking(:,2),7)-complexity(chunking(:,2),8)],[ICRT_correct_chunk_Ns6(:,3)-ICRT_correct_chunk_Ns6(:,4)],'.','MarkerSize',20); lsline
        %xlabel('\Delta Complexity (bits)'); ylabel('\Delta ICRT (ms)')

        % for all trials
        %ICRT_correct_chunk_Ns4 = ICRT_all(chunking(:,1),1:4);   % get the corresponding test RTs
        %ICRT_correct_noChunk_Ns4 = ICRT_all(~chunking(:,1),1:4);
        %ICRT_correct_chunk_Ns6 = ICRT_all(chunking(:,2),5:8);   % get the corresponding test RTs
        %ICRT_correct_noChunk_Ns6 = ICRT_all(~chunking(:,2),5:8);


        nSubjChunk = [size(ICRT_correct_chunk_Ns4,1) size(ICRT_correct_chunk_Ns4,1)  size(ICRT_correct_chunk_Ns6,1)  size(ICRT_correct_chunk_Ns6,1)]
        nSubjnoChunk = [size(ICRT_correct_noChunk_Ns4,1) size(ICRT_correct_noChunk_Ns4,1)  size(ICRT_correct_noChunk_Ns6,1)  size(ICRT_correct_noChunk_Ns6,1)]

        % ICRT analysis -  4 panels
        figure; hold on;
        subplot 221; hold on;
        colororder(gca,cmap(1:2,:));
        tmp = [nanmean(ICRT_correct_chunk_Ns4(:,1:2),1) nanmean(ICRT_correct_chunk_Ns6(:,1:2),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_correct_chunk_Ns4(:,1:2),[],1) nanstd(ICRT_correct_chunk_Ns6(:,1:2),[],1)]./sqrt(nSubjChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        title('Chunkers')
        ylim([400 1000])

        subplot 222; hold on;
        colororder(gca,cmap(1:2,:));
        tmp = [nanmean(ICRT_correct_noChunk_Ns4(:,1:2),1) nanmean(ICRT_correct_noChunk_Ns6(:,1:2),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_correct_noChunk_Ns4(:,1:2),[],1) nanstd(ICRT_correct_noChunk_Ns6(:,1:2),[],1)]./sqrt(nSubjnoChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        title('Non-Chunkers')
        ylim([400 1000])

        subplot 223; hold on;
        colororder(gca,cmap(3:4,:));
        tmp = [nanmean(ICRT_correct_chunk_Ns4(:,3:4),1) nanmean(ICRT_correct_chunk_Ns6(:,3:4),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_correct_chunk_Ns4(:,3:4),[],1) nanstd(ICRT_correct_chunk_Ns6(:,3:4),[],1)]./sqrt(nSubjChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Structured Test', 'Random Test', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        ylim([400 1000])


        subplot 224; hold on;
        colororder(gca,cmap(3:4,:));
        tmp = [nanmean(ICRT_correct_noChunk_Ns4(:,3:4),1) nanmean(ICRT_correct_noChunk_Ns6(:,3:4),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_correct_noChunk_Ns4(:,3:4),[],1) nanstd(ICRT_correct_noChunk_Ns6(:,3:4),[],1)]./sqrt(nSubjnoChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Structured Test', 'Random Test', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        ylim([400 1000])

        set(gcf, 'Position',  [400, 400, 660, 470])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_1.pdf'], 'ContentType', 'vector');

        % Chunks lead to lower RTs in Structured?
        [h,p,~,stats] = ttest(ICRT_correct(:,1)-ICRT_correct(:,2),ICRT_correct(:,5)-ICRT_correct(:,6));
        disp('Chunks lead to lower RTs in Structured Train in Ns = 6 vs. Ns = 4? (Only correct trials)')
        disp(['t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        [h,p,~,stats] = ttest2(ICRT_correct_chunk_Ns4(:,1)-ICRT_correct_chunk_Ns4(:,2),ICRT_correct_chunk_Ns6(:,1)-ICRT_correct_chunk_Ns6(:,2));
        disp('Chunks lead to lower RTs in Structured Train in Ns = 6 vs. Ns = 4? (Only correct trials in participants who chunked)')
        disp(['t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        % Decrease in ICRT from Random-> Structured Train? (Chunkers)
        [h,p,~,stats] = ttest(ICRT_correct_chunk_Ns4(:,1),ICRT_correct_chunk_Ns4(:,2));
        disp('Decrease in ICRT Train (Chunkers - Correct Trials)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,~,stats] = ttest(ICRT_correct_chunk_Ns6(:,1),ICRT_correct_chunk_Ns6(:,2));
        disp(['Ns6:t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        % Increase in ICRT from Random-> Structured Train? (NonChunkers)
        [h,p,~,stats] = ttest(ICRT_correct_noChunk_Ns4(:,1),ICRT_correct_noChunk_Ns4(:,2));
        disp('Increase in ICRT Train (NonChunkers - Correct Trials)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,~,stats] = ttest(ICRT_correct_noChunk_Ns6(:,1),ICRT_correct_noChunk_Ns6(:,2));
        disp(['Ns6:t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        % Increase in ICRT from Structured-> Random Test?
        [h,p,~,stats] = ttest(ICRT_correct_chunk_Ns4(:,4),ICRT_correct_chunk_Ns4(:,3));
        disp('Increase in ICRT in Ns = 4 after breaking chunks? (Only correct trials in participants who chunked)')
        disp(['t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,~,stats] = ttest(ICRT_correct_chunk_Ns6(:,4),ICRT_correct_chunk_Ns6(:,3));
        disp('Increase in ICRT in Ns = 6 after breaking chunks? (Only correct trials in participants who chunked)')
        disp(['t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])


        % Nonchunkers increase in ICRT from Structured-> Random Train?
        [h,p,~,stats] = ttest(ICRT_correct_noChunk_Ns4(:,4),ICRT_correct_noChunk_Ns4(:,3));
        disp('Increase in ICRT in Ns = 4 after breaking chunks? (Only correct trials in nonChunkers)')
        disp(['t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,~,stats] = ttest(ICRT_correct_noChunk_Ns6(:,4),ICRT_correct_noChunk_Ns6(:,3));
        disp('Increase in ICRT in Ns = 6 after breaking chunks? (Only correct trials in nonChunkers)')
        disp(['t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        % More of an increase (inhibition) in Test phase in Ns6 (bar
        % graphs) than in Ns4 (chunk breaking)
        %[h,p] = ttest2(ICRT_correct_chunk_Ns4(:,4)-ICRT_correct_chunk_Ns4(:,3),ICRT_correct_chunk_Ns6(:,4)-ICRT_correct_chunk_Ns6(:,3));
        %disp('More increase in ICRT in Ns = 6 than Ns = 4 after breaking chunks? (Only correct trials in participants who chunked)')
        %disp(strcat('Reject null?:',num2str(h))); disp(strcat('p-value:',num2str(p)));

    case 'ICRT_incorrect_chunk'
        conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns6,random_train', 'Ns6,structured_train'};

        chunking(:,1) = ICRT_all(:,1)-ICRT_all(:,2)>0;              % Ns4: chunk n = 57 | noChunk n = 19
        chunking(:,2) = ICRT_all(:,5)-ICRT_all(:,6)>0;              % Ns6: chunk n = 53 | noChunk n = 23

        ICRT_incorrect_chunk_Ns4 = ICRT_incorrect(chunking(:,1),1:4);   % get the corresponding test RTs
        ICRT_incorrect_noChunk_Ns4 = ICRT_incorrect(~chunking(:,1),1:4);
        ICRT_incorrect_chunk_Ns6 = ICRT_incorrect(chunking(:,2),5:8);   % get the corresponding test RTs
        ICRT_incorrect_noChunk_Ns6 = ICRT_incorrect(~chunking(:,2),5:8);

        nSubjChunk = [size(ICRT_incorrect_chunk_Ns4,1) size(ICRT_incorrect_chunk_Ns4,1)  size(ICRT_incorrect_chunk_Ns6,1)  size(ICRT_incorrect_chunk_Ns6,1)]
        nSubjnoChunk = [size(ICRT_incorrect_noChunk_Ns4,1) size(ICRT_incorrect_noChunk_Ns4,1)  size(ICRT_incorrect_noChunk_Ns6,1)  size(ICRT_incorrect_noChunk_Ns6,1)]

        % ICRT analysis -  4 panels
        figure; hold on;
        subplot 221; hold on;
        colororder(gca,cmap(1:2,:));
        tmp = [nanmean(ICRT_incorrect_chunk_Ns4(:,1:2),1) nanmean(ICRT_incorrect_chunk_Ns6(:,1:2),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_incorrect_chunk_Ns4(:,1:2),[],1) nanstd(ICRT_incorrect_chunk_Ns6(:,1:2),[],1)]./sqrt(nSubjChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        title('Chunkers')
        ylim([400 1000])

        subplot 222; hold on;
        colororder(gca,cmap(1:2,:));
        tmp = [nanmean(ICRT_incorrect_noChunk_Ns4(:,1:2),1) nanmean(ICRT_incorrect_noChunk_Ns6(:,1:2),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_incorrect_noChunk_Ns4(:,1:2),[],1) nanstd(ICRT_incorrect_noChunk_Ns6(:,1:2),[],1)]./sqrt(nSubjnoChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Random Train', 'Structured Train', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        title('Non-Chunkers')
        ylim([400 1000])

        subplot 223; hold on;
        colororder(gca,cmap(3:4,:));
        tmp = [nanmean(ICRT_incorrect_chunk_Ns4(:,3:4),1) nanmean(ICRT_incorrect_chunk_Ns6(:,3:4),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_incorrect_chunk_Ns4(:,3:4),[],1) nanstd(ICRT_incorrect_chunk_Ns6(:,3:4),[],1)]./sqrt(nSubjChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Structured Test', 'Random Test', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        ylim([400 1000])


        subplot 224; hold on;
        colororder(gca,cmap(3:4,:));
        tmp = [nanmean(ICRT_incorrect_noChunk_Ns4(:,3:4),1) nanmean(ICRT_incorrect_noChunk_Ns6(:,3:4),1)]; plotICRT(1,:) = tmp(1:length(conds)/2); plotICRT(2,:) = tmp(length(conds)/2+1:length(conds));
        se = [nanstd(ICRT_incorrect_noChunk_Ns4(:,3:4),[],1) nanstd(ICRT_incorrect_noChunk_Ns6(:,3:4),[],1)]./sqrt(nSubjnoChunk); se = transpose(reshape(se, [length(conds)/2 2]));
        b = bar([1 2], plotICRT);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', plotICRT, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        legend('Structured Test', 'Random Test', 'Location', 'northwest'); legend('boxoff');
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        xlabel('Set size'); ylabel('ICRT (ms)');
        ylim([400 1000])

        set(gcf, 'Position',  [400, 400, 660, 470])

    case 'corr_RT_train_test'
        chunking(:,1) = ICRT_all(:,1)-ICRT_all(:,2)>0;              % Ns4: chunk n = 57 | noChunk n = 19
        ICRT_correct_chunk_Ns4 = ICRT_correct(chunking(:,1),1:4);   % get the corresponding test RTs
        ICRT_correct_noChunk_Ns4 = ICRT_correct(~chunking(:,1),1:4);
        nSubjNs4 = sum(chunking(:,1));

        chunking(:,2) = ICRT_all(:,5)-ICRT_all(:,6)>0;              % Ns6: chunk n = 53 | noChunk n = 23
        ICRT_correct_chunk_Ns6 = ICRT_correct(chunking(:,2),5:8);   % get the corresponding test RTs
        ICRT_correct_noChunk_Ns6 = ICRT_correct(~chunking(:,2),5:8);

        % The level of chunking (reduction) predicts how much inhibition happens in Ns = 6 but not Ns = 4
        figure; hold on;
        x = [ICRT_correct_chunk_Ns4(:,1)-ICRT_correct_chunk_Ns4(:,2)];
        y = [ICRT_correct_chunk_Ns4(:,4)-ICRT_correct_chunk_Ns4(:,3)];
        [r,p] = corr(x,y, 'rows','complete');
        % ci = bootci(5000,@corr,x,y);

        disp('Does level of chunking predict response inhibition in random test?')
        disp(strcat('Ns4: r=',num2str(r))); disp(strcat('p=',num2str(p)));

        x2 = [ICRT_correct_chunk_Ns6(:,1)-ICRT_correct_chunk_Ns6(:,2)];
        y2 = [ICRT_correct_chunk_Ns6(:,4)-ICRT_correct_chunk_Ns6(:,3)];
        [r,p] = corr(x2,y2, 'rows','complete');
        %x2(18) = []; y2(18) = [];
        % ci = bootci(1000,@corr,x2,y2)

        disp(strcat('Ns6: r=',num2str(r))); disp(strcat('p=',num2str(p)));

        plot(x,y,'.','Color',[0.5 0.5 0.5],'MarkerSize',20);
        plot(x2,y2,'k.','MarkerSize',20);
        xlabel('Decrease in ICRT: Train (ms)');
        ylabel('Increase in ICRT: Test (ms)')
        legend('Ns = 4', 'Ns = 6')
        axis([-200 700 -400 400])
        axis square
        lsline;
        title('Chunkers - Correct ICRTs')
        set(gcf, 'Position',  [400, 400, 400, 365])

        % linear mixed effects
        %         setsize = [4*ones(1,length(y)), 6*ones(1,length(y2))];
        %         y = [y; y2];   % increase in ICRT
        %         x = [x; x2];  % decrease in ICRT
        %         tbl = table;
        %         tbl.y = y(:);
        %         tbl.x = x(:);
        %         tbl.setsize = categorical(setsize(:));
        %         lme = fitlme(tbl,'y ~ x*setsize');
        %         y_fit = fitted(lme); y_fit = reshape(y_fit,trials,n);

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

        % all ICRT trials
        figure; hold on;
        x = [ICRT_all(:,1)-ICRT_all(:,2)];
        y = [ICRT_all(:,4)-ICRT_all(:,3)];
        [r,p] = corr(x,y, 'rows','complete');
        disp('Does level of chunking predict response inhibition in random test?')
        disp(strcat('Ns4: r=',num2str(r))); disp(strcat('p=',num2str(p)));

        x2 = [ICRT_all(:,5)-ICRT_all(:,6)];
        y2 = [ICRT_all(:,7)-ICRT_all(:,8)];
        [r,p] = corr(x2,y2, 'rows','complete');
        disp(strcat('Ns6: r=',num2str(r))); disp(strcat('p=',num2str(p)));

        plot(x,y,'.','Color',[0.5 0.5 0.5],'MarkerSize',20);
        plot(x2,y2,'k.','MarkerSize',20);
        xlabel('Decrease in ICRT: Train (ms)');
        ylabel('Increase in ICRT: Test (ms)')
        legend('Ns = 4', 'Ns = 6')
        axis([-200 700 -400 400])
        axis square
        lsline;

        title('All Subjects - All ICRTs')

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_all.pdf'], 'ContentType', 'vector');
    case 'ICRT_complexity'
        load 'chunking_data.mat'
        ICRT_correct_chunk_Ns4 = ICRT_correct(chunking(:,1),1:4);   % get the corresponding test RTs
        ICRT_correct_noChunk_Ns4 = ICRT_correct(~chunking(:,1),1:4);
        ICRT_correct_chunk_Ns6 = ICRT_correct(chunking(:,2),5:8);   % get the corresponding test RTs
        ICRT_correct_noChunk_Ns6 = ICRT_correct(~chunking(:,2),5:8);

        cost_correct_chunk_Ns4 = cost_correct(chunking(:,1),1:4);
        cost_correct_noChunk_Ns4 = cost_correct(~chunking(:,1),1:4);
        cost_correct_chunk_Ns6 = cost_correct(chunking(:,2),5:8);
        cost_correct_noChunk_Ns6 = cost_correct(~chunking(:,2),5:8);

        cost_chunk_Ns4 = COST(chunking(:,1),1:4);
        cost_noChunk_Ns4 = COST(~chunking(:,1),1:4);
        cost_chunk_Ns6 = COST(chunking(:,2),5:8);
        cost_noChunk_Ns6 = COST(~chunking(:,2),5:8);

        % only when running Chunk model (4): chunkers
        if data(1).chunk == 1
            figure; hold on;subplot 211; hold on;
            plot([cost_correct_chunk_Ns4(:,1)-cost_correct_chunk_Ns4(:,2)],[ICRT_correct_chunk_Ns4(:,1)-ICRT_correct_chunk_Ns4(:,2)],'.','Color',cmap(1,:),'MarkerSize',20);
            plot([cost_correct_chunk_Ns4(:,3)-cost_correct_chunk_Ns4(:,4)],[ICRT_correct_chunk_Ns4(:,3)-ICRT_correct_chunk_Ns4(:,4)],'.','Color',cmap(2,:),'MarkerSize',20);
            xlabel('\Delta Complexity');ylabel('\Delta ICRT (ms)')
            xlim([-2 2]); ylim([-800 800]); l(1:2)=lsline;
            subplot 212; hold on;colororder(cmap(3:4,:))
            plot([cost_correct_chunk_Ns6(:,1)-cost_correct_chunk_Ns6(:,2)],[ICRT_correct_chunk_Ns6(:,1)-ICRT_correct_chunk_Ns6(:,2)],'.','Color',cmap(3,:),'MarkerSize',20);
            plot([cost_correct_chunk_Ns6(:,3)-cost_correct_chunk_Ns6(:,4)],[ICRT_correct_chunk_Ns6(:,3)-ICRT_correct_chunk_Ns6(:,4)],'.','Color',cmap(4,:),'MarkerSize',20);
            xlabel('\Delta Complexity');ylabel('\Delta ICRT (ms)')
            xlim([-2 2]); ylim([-800 800]); l(3:4)=lsline;
            legend(l,'Random Train','Structured Train','Structured Test', 'Random Test')
            sgtitle('Chunk Model')
            set(gcf, 'Position',  [400, 400, 400, 600])

            exportgraphics(gcf,[pwd '/figures/raw/' sim 'ICRT_complexity_chunkers.pdf'], 'ContentType', 'vector');

        else
            figure; hold on; subplot 211; hold on;
            plot([cost_correct_noChunk_Ns4(:,1)-cost_correct_noChunk_Ns4(:,2)],[ICRT_correct_noChunk_Ns4(:,1)-ICRT_correct_noChunk_Ns4(:,2)],'.','Color',cmap(1,:),'MarkerSize',20);
            plot([cost_correct_noChunk_Ns4(:,3)-cost_correct_noChunk_Ns4(:,4)],[ICRT_correct_noChunk_Ns4(:,3)-ICRT_correct_noChunk_Ns4(:,4)],'.','Color',cmap(2,:),'MarkerSize',20);
            xlabel('\Delta Complexity');ylabel('\Delta ICRT (ms)')
            xlim([-2 2]); ylim([-200 200]); l(1:2)=lsline; ylim([-200 200]);
            subplot 212; hold on;
            plot([cost_correct_noChunk_Ns6(:,1)-cost_correct_noChunk_Ns6(:,2)],[ICRT_correct_noChunk_Ns6(:,1)-ICRT_correct_noChunk_Ns6(:,2)],'.','Color',cmap(3,:),'MarkerSize',20);
            plot([cost_correct_noChunk_Ns6(:,3)-cost_correct_noChunk_Ns6(:,4)],[ICRT_correct_noChunk_Ns6(:,3)-ICRT_correct_noChunk_Ns6(:,4)],'.','Color',cmap(4,:),'MarkerSize',20);
            xlabel('\Delta Complexity');ylabel('\Delta ICRT (ms)')
            xlim([-2 2]); ylim([-200 200]); l(3:4)=lsline; ylim([-200 200]);
            legend(l,'Random Train','Structured Train','Structured Test', 'Random Test')
            sgtitle('NoChunk Model')
            set(gcf, 'Position',  [400, 400, 400, 600])
            exportgraphics(gcf,[pwd '/figures/raw/' sim 'ICRT_complexity_nonChunkers.pdf'], 'ContentType', 'vector');
        end


    case 'ICRT_vs_NCRT'

        % all trials
        figure;  subplot 211; hold on;
        h = barwitherr([sem(ICRT_all,1); sem(NCRT_all,1)]', [nanmean(ICRT_all);nanmean(NCRT_all)]');

        ylabel('Response Time (ms)')
        legend('ICRT','Non-ICRT')
        ylim([400 900])
        title('All Trials')

        % correct trials
        subplot 212; hold on;
        h = barwitherr([sem(ICRT_correct,1); sem(NCRT_correct,1)]', [nanmean(ICRT_correct);nanmean(NCRT_correct)]');

        ylabel('Response Time (ms)')
        legend('ICRT','Non-ICRT')
        ylim([400 900])
        set(gca, 'XTick',1:8, 'XTickLabel', {'Ns=4, Random Train','Ns=4, Structured Train','Ns=4, Structured Test', 'Ns=4, Random Test',...
            'Ns=6, Random Train','Ns=6, Structured Train','Ns=6, Structured Test', 'Ns=6, Random Test'}); box off;
        xtickangle(45)
        title('Correct Trials')


        %figure;hold on;plot(ICRT_trial(1,:,4),'.-')
        %plot(NCRT_trial(1,:,4),'.-')



   

        %subplot 323; hold on; ylim([500 1000])
        % ICRTs changing over time (correct only)
        %h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,1)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        %h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,2)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        %xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Correct Trials')

        %subplot 325; hold on;  ylim([400 1400])
        % ICRTs changing over time (incorrect only)
        %shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,1)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        %shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,2)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        %xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Incorrect Trials')
        %sgtitle('Ns = 4 Train','FontSize',25)

        %         subplot 322; hold on; ylim([500 1000])
        %         h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,3)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        %         h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,4)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        %         xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - All Trials')
        %         legend([h(1).mainLine h(2).mainLine],{'Structured Test','Random Test'}); legend('boxoff')
        %
        %         subplot 324; hold on; ylim([500 1000])
        %         % ICRTs changing over time (correct only)
        %         h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,3)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        %         h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,4)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        %         xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Correct Trials')
        %
        %         subplot 326; hold on; ylim([400 1400])
        %         % ICRTs changing over time (incorrect only)
        %         shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,3)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        %         shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,4)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        %         xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Incorrect Trials')
        %         sgtitle('Ns = 4 Test','FontSize',25)

        %exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

    case 'ICRT_trials_Ns6'
        cmap =[238 123 100 % Ns4 Random
            118 181 197
            216 38 0 % Ns6 Random
            30 129 176] / 255;
        nTrials = 20;
        ICRT_all_trials = nan(nSubj, nTrials, length(conds));
        ICRT_correct_trials = nan(nSubj, nTrials, length(conds));
        ICRT_incorr_trials = nan(nSubj, nTrials, length(conds));
        chunkInit = [2,5]; % chunk-initiating state is 2 for Ns = 4 and 5 for Ns = 6
        for s = 1:nSubj
            for c = 1:length(conds)
                idx = strcmp(data(s).cond, conds(c));
                state = data(s).s(idx);
                action = data(s).a(idx);
                reward = data(s).r(idx);
                rt = data(s).rt(idx);
                if contains(conds(c),'4')
                    condIdx = 1;
                elseif contains(conds(c), '6')
                    condIdx = 2;
                end
                ics = find(state==chunkInit(condIdx))+1; ics(ics>length(state))=[]; % intra-chunk state
                ICRT_all_trials(s,1:length(ics),c) = rt(ics);                       % 8 'pages', one for each condition
                accuracy_ics(s,1:length(ics),c) = reward(ics);
                %corr_chunk = rt(intersect(find(state == action), ics));
                %ICRT_correct_trials(s,1:length(corr_chunk),c) = corr_chunk; % correct ICRTs
                %incorr_chunk = rt(intersect(find(state ~= action), ics));
                %ICRT_incorr_trials(s,1:length(incorr_chunk),c) = incorr_chunk; % incorrect ICRTs
            end
        end

        ICRT_correct_trials = ICRT_all_trials;
        ICRT_correct_trials(accuracy_ics==0) = NaN;

        ICRT_incorr_trials = ICRT_all_trials;
        ICRT_incorr_trials(accuracy_ics==1) = NaN;

        chunking(:,1) = ICRT_all(:,1)-ICRT_all(:,2)>0;              % Ns
        chunking(:,2) = ICRT_all(:,5)-ICRT_all(:,6)>0;


        ICRT_correct_trials = ICRT_correct_trials(chunking(:,2),:,5:8);
        ICRT_incorr_trials = ICRT_incorr_trials(chunking(:,2),:,5:8);

        % ICRTs changing over time (all)
        % took mean of all the ICRTs and computed their moving average
        % across time

        figure; hold on; win = 5;
        subplot 321; hold on; ylim([500 1000])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,5)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,5),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,6)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,6),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - All Trials')
        legend([h(1).mainLine h(2).mainLine],{'Random Train','Structured Train'}); legend('boxoff')

        subplot 323; hold on; ylim([500 1000])
        % ICRTs changing over time (correct only)
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,1)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,2)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Correct Trials')

        subplot 325; hold on;  ylim([400 1400])
        % ICRTs changing over time (incorrect only)
        shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,1)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,2)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Incorrect Trials')
        sgtitle('Ns = 6 Train','FontSize',25)
        set(gcf, 'Position',  [800, 100, 350, 700])


        subplot 322; hold on; ylim([500 1000])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,7)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,7),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,8)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,8),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - All Trials')
        legend([h(1).mainLine h(2).mainLine],{'Structured Test','Random Test'}); legend('boxoff')

        subplot 324; hold on; ylim([500 1000])
        % ICRTs changing over time (correct only)
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,3)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_correct_trials(:,:,4)),win,'omitnan'), movmean(sem(ICRT_correct_trials(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Correct Trials')

        subplot 326; hold on;  ylim([400 1400])
        % ICRTs changing over time (incorrect only)
        shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,3)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_incorr_trials(:,:,4)),win,'omitnan'), movmean(sem(ICRT_incorr_trials(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('ICRT (ms)'); title('ICRTs - Incorrect Trials')
        sgtitle('Ns = 6 Test','FontSize',25)
        set(gcf, 'Position',  [1200, 100, 600, 700])

        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

    case 'actionSlips'
        actionSlip = zeros(nSubj, 2); reps = zeros(nSubj, 2);
        slipCond = {'Ns4,random_test', 'Ns6,random_test'};

        %slipCond = {'Ns4,structured_test', 'Ns6,structured_test'};
        slipPos = [2,5];
        chunkResp = [1,4];
        for s = 1:nSubj
            for c = 1:2
                state = data(s).s(strcmp(data(s).cond, slipCond(c)));
                action = data(s).a(strcmp(data(s).cond, slipCond(c)));
                rt = data(s).rt(strcmp(data(s).cond, slipCond(c)));
                ics = find(state==slipPos(c))+1; % find the intrachunk state
                ics(ics>length(state))=[];
                actionSlip(s,c) = sum(state(ics)~=chunkResp(c) & action(ics)==chunkResp(c)); % it's only an action slip if you took the old intrachunk action in what would have been an intrachunk state (but was not)
                reps(s,c) = sum(state(ics)~=chunkResp(c)); % how many times a non-intrachunk state appeared (have to normalize by this)
                actionCorrect(s,c) = sum(state(ics)==chunkResp(c) & action(ics)==chunkResp(c));
                corr_reps(s,c) = sum(state(ics)==chunkResp(c)); % how many times a intrachunk state appeared (have to normalize by this)
                rt_slip(s,c) = nanmean(rt(state(ics)~=chunkResp(c) & action(ics)==chunkResp(c))); % rt of the action slip
                rt_chunks(s,c) = nanmean(rt(state(ics)==chunkResp(c) & action(ics)==chunkResp(c))); %
                rt_nonchunk_incorrect(s,c) = nanmean(rt(state(ics)~=chunkResp(c) & action(ics)~=chunkResp(c) & state(ics)~=action(ics))); % not an action slip, but also incorrect
                rt_nonchunk_correct(s,c) = nanmean(rt(state(ics)~=chunkResp(c)  & action(ics)~=chunkResp(c) & state(ics)==action(ics))); % correct actions that are not the action chunk

            end
        end

        figure; hold on; colororder(cmap(4,:));
        subplot 121; hold on;
        %actionSlip(actionSlip==0) = NaN; % normalized slips
        b = bar(1:2, mean(actionSlip./reps,1));
        errorbar(1:2, mean(actionSlip./reps,1), sem(actionSlip./reps,1), sem(actionSlip./reps,1), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        xtips = b.XEndPoints;
        ytips = b.YEndPoints;
        labels = string(b.YData);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4, Test', 'Ns=6, Test'}); box off;
        ylabel('Percentage of Action Slips')

        subplot 122; hold on;
        b = bar(1:4, [nanmean(rt_slip,1); nanmean(rt_chunks,1); nanmean(rt_nonchunk_incorrect,1); nanmean(rt_nonchunk_correct,1)]);
        %errorbar(1:2, nanmean(rt_slip,1), sem(rt_slip,1), sem(rt_slip,1), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        errorbar([1:4; 1:4]', [nanmean(rt_slip,1); nanmean(rt_chunks,1); nanmean(rt_nonchunk_incorrect,1); nanmean(rt_nonchunk_correct,1)], [sem(rt_slip,1); sem(rt_chunks,1); sem(rt_nonchunk_incorrect,1); sem(rt_nonchunk_correct,1)], [sem(rt_slip,1); sem(rt_chunks,1); sem(rt_nonchunk_incorrect,1); sem(rt_nonchunk_correct,1)], 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);

        set(gca, 'XTick',1:4, 'XTickLabel', {'RT slip', 'RT chunks', 'RT nonchunk incorrect','RT nonchunk correct'}); box off;
        xtickangle(45)
        ylabel('RT of action slips')



        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '.pdf'], 'ContentType', 'vector');

end


end