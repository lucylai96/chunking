function analysis_policy_complexity(plotCase, data, recode)

%{
    Analyses and plottings related to policy complexity, including average
    policy complexity in different blocks, reward-complexity curves,
    and rain cloud plot of policy complexity distribution in different blocks.

    USAGE:
        analysis_policy_complexity('avgComplexity', data)
        analysis_policy_complexity('avgComplexity_train', data)
        analysis_policy_complexity('avgComplexity_test', data)
        analysis_policy_complexity('RT_complexity', data)
        analysis_policy_complexity('RC_curves', data)
        analysis_policy_complexity('RC_empirical_curves', data)
        analysis_policy_complexity('RC_bars', data)*
        
        analysis_policy_complexity('complexity_trials', data)
        analysis_policy_complexity('RC_curves_no_scatter', data)
        analysis_policy_complexity('change_ICRT_complexity', data)
    
    Input: 'plotCase' is a string representing which plot to show; data is
    an optional input and can be the simulated data structure

    Called by: plot_all_figures()
%}


%if nargin<2;load('actionChunk_data.mat',daata,'R_compl); else sim='sim_';end
sim = [];

cmap =[141 182 205
    255 140 105
    238 201 0
    155 205 155] / 255;

nSubj = length(data);
conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns4,structured_test','Ns4,random_test',...
    'Ns6,random_train', 'Ns6,structured_train', 'Ns6,structured_test','Ns6,random_test'};
if nargin>1
    %all = specs.all; chunkers = specs.chunkers; recode = specs.recode; % Use recoded states to calculate policy complexity
    all = 1; chunkers = 0; %recode = 1;
else
    all = 1; chunkers = 0; %recode = 1; % Use recoded states to calculate policy complexity
end

%recode = 1;

maxreward = [80 80 60 60 120 120 90 90]; chunk_only = 0;
[reward, complexity] = calculateRPC(data, conds, recode, maxreward, chunk_only);


chunkInit = [2,5];
if isfield(data(1),'rt')
    for s = 1:nSubj
        for c = 1:length(conds)
            idx = strcmp(data(s).cond, conds(c));
            state = data(s).s(idx);
            action = data(s).a(idx);
            rt = data(s).rt(idx);
        end
    end
    %Calculate average RT / ICRT
    avgRT = nan(nSubj, length(conds));
    ICRT_correct = zeros(nSubj, length(conds));

    for s = 1:nSubj
        for c = 1:length(conds)
            idx = strcmp(data(s).cond, conds(c));
            state = data(s).s(idx);
            action = data(s).a(idx);
            r = data(s).r(idx);
            rt = data(s).rt(idx);
            avgRT(s,c) = nanmean(rt);

            for i = 1:max(state)
                ps = sum(state==i)/length(state);
                for j = 1:max(action)
                    pas(j) = sum(action(state==i)==j)/sum(state==i);
                end
                pas(pas<0.001) = 0.001;
                has(i) = sum(ps.*pas.*log2(pas));
            end
            entropy(s,c) = -sum(has);

            if contains(conds(c),'4'); condIdx = 1; end
            if contains(conds(c),'6'); condIdx = 2; end
            cis = find(state==chunkInit(condIdx));
            ics = cis+1; ics(ics>length(state))=[];
            cs = sort([cis; ics]);
            ICRT_all(s,c) = nanmean(rt(ics));
            ICRT_correct(s,c) = nanmean(rt(intersect(find(state == action), ics)));
            if sim % if it's a simulation
                %cost = data(s).ecost(idx);
                complexity(s,c) = mean(data(s).cost(idx)); % model-generated cost (ics)
                %complexity(s,c) = mean(cost(ics)); % model-generated cost (ics)
                %reward(s,c) = mean(r(ics)); % model-generated cost (ics)
            end
        end
    end
end % if real data


%chunking(:,1) = ICRT_all(:,1)-ICRT_all(:,2)>0;              % Ns4: chunk n = 57 | noChunk n = 19
%chunking(:,2) = ICRT_all(:,5)-ICRT_all(:,6)>0;              % Ns4: chunk n = 57 | noChunk n = 19
if ~all
    load('chunking_data.mat')
    if chunkers == 1
        complexity(chunking(:,1)==0,1:4) = NaN; % only chunkers
        reward(chunking(:,1)==0,1:4) = NaN; % only chunkers
        complexity(chunking(:,2)==0,5:8) = NaN; % only chunkers
        reward(chunking(:,2)==0,5:8) = NaN; % only chunkers
        nSubj = sum(~isnan(complexity(:,1:8)));
    else
        complexity(chunking(:,1)==1,1:4) = NaN; % only nonchunkers
        reward(chunking(:,1)==1,1:4) = NaN; % only nonchunkers
        complexity(chunking(:,2)==1,5:8) = NaN; % only nonchunkers
        reward(chunking(:,2)==1,5:8) = NaN; % only nonchunkers
        nSubj = sum(~isnan(complexity(:,1:8)));
    end
end
% Calculate average policy complexity for different blocks
avgComplx = reshape(nanmean(complexity,1), [4 2])';
se = nanstd(complexity,1)./sqrt(nSubj);
se = reshape(se, [4 2])';


switch plotCase
    %% Plot average policy complexity for different blocks
    case 'avgComplexity'
        figure; hold on;
        colororder(cmap);

        b = bar(avgComplx);
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', avgComplx, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        ylabel('Policy Complexity');
        legend('Random Train', 'Structured Train','Structured Test', 'Random Test','Location', 'northwest'); legend('boxoff');
        ylim([0 1.5]);

        set(gcf, 'Position',  [400, 400, 650, 300])

    case 'avgComplexity_train'
        figure; hold on;
        colororder(cmap);

        b = bar(avgComplx(:,1:2));
        errorbar_pos = errorbarPosition(b, se(:,1:2));
        errorbar(errorbar_pos', avgComplx(:,1:2), se(:,1:2), se(:,1:2), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        ylabel('Policy Complexity'); ylim([0 1.5]);
        legend('Random Train','Structured Train','Location', 'northwest'); legend('boxoff');

        set(gcf, 'Position',  [400, 400, 500, 270])


    case 'avgComplexity_test'
        figure; hold on;
        colororder(cmap(3:4,:));
        b = bar(avgComplx(:,3:4));
        errorbar_pos = errorbarPosition(b, se(:,3:4));
        errorbar(errorbar_pos', avgComplx(:,3:4), se(:,3:4), se(:,3:4), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'}); ylim([0 1.5]);
        legend('Structured Test', 'Random Test','Location', 'northwest'); legend('boxoff');
        ylabel('Policy Complexity');

        set(gcf, 'Position',  [400, 400, 500, 270])


    case 'RT_complexity'
        maxreward = [80 80 60 60 120 120 90 90]; chunk_only = 0;
        avgRT = log(avgRT);


        %         bin = 5;
        %         figure; subplot 221; hold on; colororder(cmap(3:4,:))
        %         [x,y,se,X] = quantile_stats_x(complexity(:,3),avgRT(:,3),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %
        %         [x,y,se,X] = quantile_stats_x(complexity(:,4),avgRT(:,4),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %         legend('Structured Test','Random Test')
        %         title('Ns=4');ylabel('RT (log ms)')
        %         xlabel('Policy complexity')
        %
        %         subplot 223; hold on; colororder(cmap(3:4,:))
        %         [x,y,se,X] = quantile_stats_x(complexity(:,7),avgRT(:,7),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %
        %         [x,y,se,X] = quantile_stats_x(complexity(:,8),avgRT(:,8),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %         legend('Structured Test','Random Test')
        %         title('Ns=6')
        %
        %
        %         recode = 0;
        %         [reward, complexity] = calculateRPC(data, conds, recode, maxreward, chunk_only);
        %         subplot 222; hold on; colororder(cmap(3:4,:))
        %         [x,y,se,X] = quantile_stats_x(complexity(:,3),avgRT(:,3),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %
        %         [x,y,se,X] = quantile_stats_x(complexity(:,4),avgRT(:,4),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %         title('Ns=4');
        %
        %         subplot 224; hold on; colororder(cmap(3:4,:))
        %         [x,y,se,X] = quantile_stats_x(complexity(:,7),avgRT(:,7),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %
        %         [x,y,se,X] = quantile_stats_x(complexity(:,8),avgRT(:,8),bin)
        %         errorbar(x,y,se/2,'.-','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        %         title('Ns=6')


        % just plot policy complexity by RT (all trials)
        figure; hold on; colororder(cmap); ylabel('RT (log ms)'); xlabel('Policy complexity')
        plot(mean(complexity),mean(avgRT),'k.');lsline;set(gca,'ColorOrderIndex',1)
        plot(complexity,avgRT,'.','MarkerSize',10); lsline
        corr(complexity(:), avgRT(:))
        for i = 1:size(complexity,2)
            plot(mean(complexity(:,i)),mean(avgRT(:,i)),'o','MarkerSize',20,'MarkerFaceColor','k')
        end
        [r,p] = corr(mean(complexity)',mean(avgRT)')
        title(strcat('All blocks: r =',num2str(r), '; p = ',num2str(p)))
        ylim([6 7.4])

        % split by train and test
        figure; hold on; idx = [1 2 5 6]; colororder(cmap(1:2,:)); ylabel('RT (log ms)'); xlabel('Policy complexity')
        plot(mean(complexity(:,idx)),mean(avgRT(:,idx)),'k.');lsline;set(gca,'ColorOrderIndex',1)
        plot(complexity(:,idx),avgRT(:,idx),'.','MarkerSize',10); lsline
        for i = idx
            plot(mean(complexity(:,i)),mean(avgRT(:,i)),'o','MarkerSize',20,'MarkerFaceColor','k')
        end
        [r,p] = corr(mean(complexity(:,idx))',mean(avgRT(:,idx))')
        title(strcat('Train: r =',num2str(r), '; p = ',num2str(p)))
        ylim([6 7.4])

        figure; hold on; idx = [3 4 7 8]; colororder(cmap(3:4,:)); ylabel('RT (log ms)'); xlabel('Policy complexity')
        plot(mean(complexity(:,idx)),mean(avgRT(:,idx)),'k.');lsline;set(gca,'ColorOrderIndex',1)
        [r,p] = corr(mean(complexity(:,idx))',mean(avgRT(:,idx))')
        plot(complexity(:,idx),avgRT(:,idx),'.','MarkerSize',10); lsline
        for i = idx
            plot(mean(complexity(:,i)),mean(avgRT(:,i)),'o','MarkerSize',20,'MarkerFaceColor','k')
        end
        title(strcat('Test: r =',num2str(r), '; p = ',num2str(p)))
        ylim([6 7.4])
        %equalabscissa(1,3)
        %         if recode == 1
        %             sgtitle('Recoded')
        %         else
        %             sgtitle('NOT Recoded')
        %         end

        % stratify by entropy
        figure; subplot 121; hold on; colororder(cmap)
        for c = 1:4
            ent = entropy(:,c);
            q = linspace(min(ent),max(ent),5);
            for i = 1:length(q)-1
                idx = ent>q(i) & ent<=q(i+1); sum(idx(:))
                x(i) = nanmean(complexity(idx));
                errx(i) = sem(complexity(idx),1);
                y(i) = nanmean(avgRT(idx));
                erry(i) = sem(avgRT(idx),1);


                %[x,y,se,X] = interval_stats_x(complexity(entropy<0.5),avgRT(entropy<0.5),5)
            end
            %plot(x,y,'.','MarkerSize',20)
            errorbar(x,y,erry/2,erry/2,errx/2,errx/2,'.','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        end
        title('Ns = 4')
        ylabel('RT (log ms)')
        xlabel('Policy complexity')

        subplot 122; hold on; colororder(cmap)
        for c = 5:8
            ent = entropy(:,c);
            q = linspace(min(ent),max(ent),7);
            for i = 1:length(q)-1
                idx = ent>q(i) & ent<=q(i+1);  sum(idx(:))
                x(i) = nanmean(complexity(idx));
                errx(i) = sem(complexity(idx),1);
                y(i) = nanmean(avgRT(idx));
                erry(i) = sem(avgRT(idx),1);

                %[x,y,se,X] = quantile_stats_x(complexity(idx),avgRT(idx),bins);
                %errorbar(x,y,se/2);
                %[x,y,se,X] = interval_stats_x(complexity(entropy<0.5),avgRT(entropy<0.5),5)
            end
            %plot(x,y,'.','MarkerSize',20)
            errorbar(x,y,erry/2,erry/2,errx/2,errx/2,'.','MarkerSize',20,'LineWidth',1.5,'Capsize',0)
        end
        title('Ns = 6')
        legend('Random Train','Structured Train','Structured Test','Random Test','Location', 'northwest'); legend('boxoff');



         % just plot policy complexity by ICRT (all trials)
         ICRT_all = log(ICRT_all);
        figure; hold on; colororder(cmap); ylabel('RT (log ms)'); xlabel('Policy complexity')
        plot(mean(complexity),mean(ICRT_all),'k.');lsline;set(gca,'ColorOrderIndex',1)
        plot(complexity,ICRT_all,'.','MarkerSize',10); lsline
        corr(complexity(:), ICRT_all(:))
        for i = 1:size(complexity,2)
            plot(mean(complexity(:,i)),mean(ICRT_all(:,i)),'o','MarkerSize',20,'MarkerFaceColor','k')
        end
        [r,p] = corr(mean(complexity)',mean(ICRT_all)')
        title(strcat('All blocks: r =',num2str(r), '; p = ',num2str(p)))
        ylim([6 7.4])

        % split by train and test
        figure; hold on; idx = [1 2 5 6]; colororder(cmap(1:2,:)); ylabel('ICRT (log ms)'); xlabel('Policy complexity')
        plot(mean(complexity(:,idx)),mean(ICRT_all(:,idx)),'k.');lsline;set(gca,'ColorOrderIndex',1)
        plot(complexity(:,idx),ICRT_all(:,idx),'.','MarkerSize',10); lsline
        for i = idx
            plot(mean(complexity(:,i)),mean(ICRT_all(:,i)),'o','MarkerSize',20,'MarkerFaceColor','k')
        end
        [r,p] = corr(mean(complexity(:,idx))',mean(ICRT_all(:,idx))')
         t1 = complexity(:,idx); t2 = ICRT_all(:,idx);
        [a,b] = corr(t1(:),t2(:))
        title(strcat('Train: r =',num2str(r), '; p = ',num2str(p)))
        ylim([6 7.4])

        figure; hold on; idx = [3 4 7 8]; colororder(cmap(3:4,:)); ylabel('ICRT (log ms)'); xlabel('Policy complexity')
        plot(mean(complexity(:,idx)),mean(ICRT_all(:,idx)),'k.');lsline;set(gca,'ColorOrderIndex',1)
        [r,p] = corr(mean(complexity(:,idx))',mean(ICRT_all(:,idx))')
        t1 = complexity(:,idx); t2 = ICRT_all(:,idx);
        [a,b] = corr(t1(:),t2(:))
        plot(complexity(:,idx),ICRT_all(:,idx),'.','MarkerSize',10); lsline
        for i = idx
            plot(mean(complexity(:,i)),mean(ICRT_all(:,i)),'o','MarkerSize',20,'MarkerFaceColor','k')
        end
        title(strcat('Test: r =',num2str(r), '; p = ',num2str(p)))
        ylim([6 7.4])



        %% Plot reward-complexity curve
    case 'RC_curves'    % Plot reward-complexity curve
        figure; hold on;
        plot_RPCcurve(reward, complexity, [1 2], {'Ns=4, Random Train', 'Ns=4, Structured Train'},'setsize',1);
        plot_RPCcurve(reward, complexity, [5 6], {'Ns=6, Random Train', 'Ns=6, Structured Train'},'setsize',1);
        plot_RPCcurve(reward, complexity, [3 4], {'Ns=4, Structured Test', 'Ns=4, Random Test'},'setsize',1);
        plot_RPCcurve(reward, complexity, [7 8], {'Ns=6, Structured Test', 'Ns=6, Random Test'},'setsize',1);

        set(gcf, 'Position',  [400, 400, 550, 500])

    case 'RC_bars'
        prettyplot
        figure; hold on;  subplot 221; hold on;
        colororder(cmap);
        b = bar(avgComplx(:,1:2));
        errorbar_pos = errorbarPosition(b, se(:,1:2));
        errorbar(errorbar_pos', avgComplx(:,1:2), se(:,1:2), se(:,1:2), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        ylabel('Policy Complexity'); ylim([0 1.5]);
        if ~sim
            plot(errorbar_pos(:,1)',complexity(:,1:2),'Color',[0.7 0.7 0.7],'LineWidth',1);
            plot(errorbar_pos(:,2)',complexity(:,5:6),'Color',[0.7 0.7 0.7],'LineWidth',1);
        end
        ylim([0.2 1.75]);
        legend('Random Train','Structured Train','Location', 'northwest'); legend('boxoff');

        subplot 223; hold on;
        avgRew = reshape(nanmean(reward,1), [4 2])';
        se = nanstd(reward,1)./sqrt(nSubj);
        se = reshape(se, [4 2])';
        b = bar(avgRew(:,1:2));
        errorbar_pos = errorbarPosition(b, se(:,1:2));
        errorbar(errorbar_pos', avgRew(:,1:2), se(:,1:2), se(:,1:2), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        ylabel('Reward'); ylim([0 1.5]);
        if ~sim
            plot(errorbar_pos(:,1)',reward(:,1:2),'Color',[0.7 0.7 0.7],'LineWidth',1);
            plot(errorbar_pos(:,2)',reward(:,5:6),'Color',[0.7 0.7 0.7],'LineWidth',1);
        end
        ylim([0.25 1])


        [h,p,ci,stats] = ttest(complexity(:,1),complexity(:,2));
        disp('Policy Complexity (Random v. Structured Train)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(complexity(:,5),complexity(:,6));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        [h,p,ci,stats] = ttest(reward(:,1),reward(:,2));
        disp('Reward (Random v. Structured Train)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(reward(:,5),reward(:,6));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        subplot 222; hold on;
        se = nanstd(complexity,1)./sqrt(nSubj);
        se = reshape(se, [4 2])';
        colororder(gca,cmap(3:4,:));
        b = bar(avgComplx(:,3:4));
        errorbar_pos = errorbarPosition(b, se(:,3:4));
        errorbar(errorbar_pos', avgComplx(:,3:4), se(:,3:4), se(:,3:4), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',3:4, 'XTickLabel', {'Ns=4', 'Ns=6'});
        ylabel('Policy Complexity'); ylim([0 1.5]);
        if ~sim
            plot(errorbar_pos(:,1)',complexity(:,3:4),'Color',[0.7 0.7 0.7],'LineWidth',1);
            plot(errorbar_pos(:,2)',complexity(:,7:8),'Color',[0.7 0.7 0.7],'LineWidth',1);
        end
        ylim([0.2 1.75])
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        legend('Structured Test','Random Test','Location', 'northwest'); legend('boxoff');

        subplot 224; hold on;
        colororder(gca,cmap(3:4,:));
        avgRew = reshape(nanmean(reward,1), [4 2])';
        se = nanstd(reward,1)./sqrt(nSubj);
        se = reshape(se, [4 2])';
        b = bar(avgRew(:,3:4));
        errorbar_pos = errorbarPosition(b, se(:,3:4));
        errorbar(errorbar_pos', avgRew(:,3:4), se(:,3:4), se(:,3:4), 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        ylabel('Reward'); ylim([0 1]);
        if ~sim
            plot(errorbar_pos(:,1)',reward(:,3:4),'Color',[0.7 0.7 0.7],'LineWidth',1);
            plot(errorbar_pos(:,2)',reward(:,7:8),'Color',[0.7 0.7 0.7],'LineWidth',1);
        end
        ylim([0.25 1])
        set(gcf, 'Position',  [400, 400, 800, 600])


        [h,p,ci,stats] = ttest(complexity(:,3),complexity(:,4));
        disp('Policy Complexity (Random v. Structured Test)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(complexity(:,7),complexity(:,8));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])

        [h,p,ci,stats] =  ttest(reward(:,3),reward(:,4));
        disp('Reward (Random v. Structured Test)')
        disp(['Ns4: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])
        [h,p,ci,stats] = ttest(reward(:,7),reward(:,8));
        disp(['Ns6: t('  num2str(stats.df) ')=' num2str(stats.tstat) ', p=' num2str(p)])


    case 'RC_empirical_curves'
        % fit empirical reward-complexity curves with polynomial
        % split into [1 vs 2], [3 vs 4]
        temp = complexity(:,3);
        complexity(:,3) = complexity(:,4);
        complexity(:,4) = temp;
        temp = reward(:,3);
        reward(:,3) = reward(:,4);
        reward(:,4) = temp;

        temp = complexity(:,7);
        complexity(:,7) = complexity(:,8);
        complexity(:,8) = temp;
        temp = reward(:,7);
        reward(:,7) = reward(:,8);
        reward(:,8) = temp;

        cond = [data.cond];
        for j = 0:1
            results.bic = zeros(4,2);
            results.aic = zeros(4,2);
            ci = 1;
            for c = 1:2:7 % ns4 train, ns4 test, ns6 train, ns6 test
                % takes only structured or random
                x = complexity(:,c+j);
                x = [ones(size(x)) x x.^2];
                y = reward(:,c+j);
                n = length(y); k = size(x,2);
                [b,bint] = regress(y,x);
                results.bci_sep(ci,j+1,:) = diff(bint,[],2)/2;
                results.b_sep(ci,j+1,:) = b;
                mse = mean((y-x*b).^2);
                results.bic(ci,1) = -2*log(mse) + k*log(n);
                results.aic(ci,1) = -2*log(mse) + k*2;
                %results.bic(ci,1) +
                % takes both structured or random
                x = complexity(:,c:c+1); x = x(:);
                x = [ones(size(x)) x x.^2];
                y = squeeze(reward(:,c:c+1)); y = y(:);
                n = length(y); k = size(x,2);
                [b,bint] = regress(y,x);
                results.bci_joint(ci,j+1,:) = diff(bint,[],2)/2;
                results.b_joint(ci,j+1,:) = b;
                mse = mean((y-x*b).^2);
                results.bic(ci,2) = -2*log(mse) + k*log(n);
                results.aic(ci,2) = -2*log(mse) + k*2;
                ci = ci+1;
            end
            if j==0;tmp = results.bic(:,1); end

        end
        % bic column 1: separate model (lower bic is better fit)
        % bic column 2: joint model
        results.bic(:,1) = mean([results.bic(:,1), tmp],2);

        titles = {'Ns=4, Train', 'Ns=4, Test','Ns=6, Train', 'Ns=6, Test'};
        fc = [1 2; 4 3; 1 2; 4 3];
        for c = 1:4
            subplot(2,2,c)
            m = squeeze(results.b_sep(c,:,:));
            err = squeeze(results.bci_sep(c,:,:));
            h = barerrorbar(m',err'); box off;
            h(1).FaceColor = cmap(fc(c,1),:);
            h(2).FaceColor = cmap(fc(c,2),:);
            %h(1).CData = C(1,:); h(2).CData = C(2,:);
            ylabel('Parameter value','FontSize',25);
            set(gca,'XTickLabel',{'\beta_0' '\beta_1' '\beta_2'},'FontSize',15,'YLim',[-8 11]);
            title(titles{c},'FontSize',20,'FontWeight','Bold');

            if c<=2
                legend(h,{'Random' 'Structured'},'FontSize',15,'Location','South');
            end
        end

        set(gcf,'Position',[200 200 500 500])
        exportgraphics(gcf,[pwd '/figures/raw/' plotCase 'params.pdf'], 'ContentType', 'vector');


        figure; hold on;
        h = bar(results.bic(:,2)-results.bic(:,1));
        set(h,'FaceColor','k')
        xticks([1:4]); xticklabels(titles); xtickangle(45);
        ylabel('\Delta BIC');

        set(gcf,'Position',[200 200 400 300])


end

if all
    if recode == 1
        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_recode_all.pdf'], 'ContentType', 'vector');
    else
        exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_all.pdf'], 'ContentType', 'vector');
    end
else
    if recode == 1
        if chunkers == 1
            exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_recode_chunkers.pdf'], 'ContentType', 'vector');
        else
            exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_recode_nonchunkers.pdf'], 'ContentType', 'vector');

        end
    else
        if chunkers == 1
            exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_chunkers.pdf'], 'ContentType', 'vector');
        else
            exportgraphics(gcf,[pwd '/figures/raw/' sim plotCase '_nonchunkers.pdf'], 'ContentType', 'vector');
        end
    end
end
end
