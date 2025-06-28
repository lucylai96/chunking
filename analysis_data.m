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

        analysis_data('ICRT_trials', data)

    Input: 'plotCase' is a string representing which plot to show; data is
    an optional input and can be the simulated data structure

    Called by: plot_all_figures()
%}

if nargin<2; load('actionChunk_data.mat'); end

nSubj = length(data);

conds = {'Ns4,random_train', 'Ns4,structured_train','Ns4,random_test', 'Ns4,structured_test',...
    'Ns6,random_train', 'Ns6,structured_train','Ns6,random_test', 'Ns6,structured_test'};
conds = {'Ns4,random_train', 'Ns4,structured_train','Ns6,random_train', 'Ns6,structured_train'};
cmap =[238 123 100 % Ns4 Random
    118 181 197
    216 38 0 % Ns6 Random
    30 129 176] / 255;

nTrials = 20; nS = [4,4,6,6]; B = [1,2,1,2]; %nS = [4,4,4,4,6,6,6,6]; B = [1,2,1,2,1,2,1,2];
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

        block_type(s,c) = B(c);
        setsize(s,c) = nS(c);
        ICRT_all_trials(s,1:length(ics),c) = rt(ics);             % 8 'pages', one for each condition
        NCRT_all_trials(s,1:length(rt(setdiff(1:end,ics))),c) = rt(setdiff(1:end,ics));
        acc_ICRT(s,1:length(ics),c) = reward(ics);          % 8 'pages', one for each condition
        acc_NCRT(s,1:length(rt(setdiff(1:end,ics))),c) = reward(setdiff(1:end,ics));         % 8 'pages', one for each condition
        err_ICRT(s,1:length(ics),c) = reward(ics)==0;          % 8 'pages', one for each condition
    end
end
sub = repmat(1:nSubj,1,size(block_type,2))';

switch plotCase

    case 'supp1'

     
        clear reward state action

        conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns6,random_train', 'Ns6,structured_train'};
        tdx = 2;
        for s = 1:nSubj
            for c = 1:length(conds)
                idx = strcmp(data(s).cond, conds(c));
                state = data(s).s(idx);
                action = data(s).a(idx);
                reward(s,c) = mean(data(s).r(idx));

                tdx = 11; alpha = 0.1;
                I_sa_s1(s,c) = cond_mutual_information(state(tdx:end),action(tdx:end),state(tdx-1:end-1),alpha); % I(S;A|S_{t-1})
                I_s_a(s,c) = mutual_information_basic(state(tdx:end),action(tdx:end),alpha); % I(S;A)
                I_a_s1(s,c) = mutual_information_basic(state(tdx-1:end-1),action(tdx:end),alpha); % I(S_{t-1};A)
                I_ss1(s,c) = mutual_information_basic(state(tdx-1:end-1),state(tdx:end),alpha); % I(S_{t-1};S)
               I_ss1_a(s,c) =  cond_mutual_information(state(tdx:end),state(tdx-1:end-1),action(tdx:end),alpha);
               
            end % cond
        end % subj

        %do people who have a higher mean(I_s_a-I_sa_s1) vs mean(I_a_s1) chunk more?
        %sav = I_s_a-I_sa_s1-I_a_s1;
      %  [I_s_a-I_sa_s1-[I_a_s1(:,1)]
Isa_s1_check = I_s_a - I_ss1 + I_ss1_a

       sav = [I_s_a(:,2)-I_sa_s1(:,2)-I_a_s1(:,2) I_s_a(:,2)-I_sa_s1(:,4)-I_a_s1(:,4)];
        %med = median(sav);
        chunkers = find(sav(:,1)>=med(1) & sav(:,2)>=med(2));
        nonchunkers = find(sav(:,1)<=med(1) & sav(:,2)<=med(2));
        
        ICRT_diff = [squeeze(nanmean(ICRT_all_trials(:,15:20,1),2)-nanmean(ICRT_all_trials(:,15:20,2),2)) squeeze(nanmean(ICRT_all_trials(:,15:20,3),2)-nanmean(ICRT_all_trials(:,15:20,4),2))];
        err_diff = [squeeze(nanmean(1-acc_ICRT(:,15:20,1),2)-nanmean(1-acc_ICRT(:,15:20,2),2)) squeeze(nanmean(1-acc_ICRT(:,15:20,3),2)-nanmean(1-acc_ICRT(:,15:20,4),2))];
       
        % stats
        tbl = table;
        tbl.icrt = ICRT_diff(:); 
        tbl.err = err_diff(:);
        %tbl.block_type = categorical(block_type(:));
        %tbl.setsize = categorical(setsize(:));
        cdiff = [I_s_a(:,2)-I_sa_s1(:,2) I_s_a(:,4)-I_sa_s1(:,4)];
        tbl.cdiff = cdiff(:);
        default = I_a_s1(:,[2,4]);
        tbl.default = default(:);
        tbl.sav = sav(:);
        tbl.sub = repmat(1:nSubj,1,2)';

        % look at the differences in complexity as a fx of block type and set size
        
        lme = fitlme(tbl, 'icrt ~ cdiff*default + (cdiff|sub)+ (default|sub)');
        anova(lme)

         lme = fitlme(tbl, 'err~ cdiff*default + (cdiff|sub)+ (default|sub)');
        anova(lme)

       % Define low, medium, and high default levels (e.g., quantiles)
lowThresh = quantile(tbl.cdiff, 0.33);
highThresh = quantile(tbl.cdiff, 0.67);

% Create cdiff group labels
tbl.cdiffGroup = repmat("medium", height(tbl), 1);
tbl.cdiffGroup(tbl.cdiff < lowThresh) = "low";
tbl.cdiffGroup(tbl.cdiff > highThresh) = "high";

% Convert to categorical
tbl.cdiffGroup = categorical(tbl.cdiffGroup);

% Plot setup
figure; hold on;
groups = categories(tbl.cdiffGroup);
colors = lines(numel(groups));

% Plot lines for each group
for i = 1:numel(groups)
    g = groups{i};
    idx = tbl.cdiffGroup == g;

    % Fit a line (or smooth) for this group
    x = tbl.default(idx);
    y = tbl.icrt(idx);

    % Optional: sort for smoother line
    [xSorted, sortIdx] = sort(x);
    ySorted = y(sortIdx);

    % Fit and plot a line
    p = polyfit(xSorted, ySorted, 1);
    yFit = polyval(p, xSorted);
    plot(xSorted, yFit, '-', 'Color', colors(i,:), 'LineWidth', 2);
    
    % Scatter data points
    scatter(x, y, 20, 'MarkerEdgeColor', colors(i,:), 'DisplayName', g);
end


ylabel('reduction in icrt');
legend('Location', 'best');
xlabel('default complexity');
title('reduction in icrt vs. default by cdiff level');



    case 'revision'
        %(1) TODO: Make a supplemental table of computed values for example sequences in Random vs Structured to compare conditional compression vs policy compression.
        %Calculations of I(S;A) and I(S;A|S_{t-1}) for each sequence
        %Show why conditional complexity decreases in Structured blocks 
        % despite an increase in standard complexity: when s1→s2 forms a predictable chunk, I(S_t;S_{t-1}) increases, which decreases I(S_t;A_t|S_{t-1}) according to our decomposition formula (Section 3 of the manuscript).


        s = 1;
        for c = 1:2
            idx = strcmp(data(s).cond, conds(c));
            state = data(s).s(idx);
            action = data(s).a(idx);action = state;
            reward(s,c) = mean(data(s).r(idx));
            state(1:10)
            
            tdx = 2; alpha = 0.1;
      
            I_sa_s1(s,c) = cond_mutual_information(state(tdx:end),action(tdx:end),state(tdx-1:end-1),alpha); % I(S;A|S_{t-1})
           
            I_s_a(s,c) = mutual_information_basic(state(tdx:end),action(tdx:end),alpha); % I(S;A)
            I_s_s1(s,c) = mutual_information_basic(state(tdx:end),state(tdx-1:end-1),alpha); % I(S;S_{t-1})

        end
        

        %(3) TODO: One way to get at this empirically would be to look at learning during the structured blocks. Compare the intrachunk RT conditional on the feedback being positive vs. negative the last time the chunk appeared. The feedback is based on correct action selection, so if they're using actions to chunk, then they should be faster after getting positive feedback, which should reinforce the chunk. A model based purely on state sequences shouldn't be influenced by feedback.

        %intrachunk RT for positive feedback vs negative feedback
        %if actions are used to chunk, then RT in next time chunk appears should be faster after getting positive feedback
        %A model based purely on state sequences shouldn't be influenced by feedback.


    case 'reward_complexity'
        clear reward state action

        conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns6,random_train', 'Ns6,structured_train'};

        for s = 1:nSubj
            for c = 1:length(conds)
                idx = strcmp(data(s).cond, conds(c));
                state = data(s).s(idx);
                action = data(s).a(idx);
                reward(s,c) = mean(data(s).r(idx));

                tdx = 11; alpha = 0.1;
                I_sa_s1(s,c) = cond_mutual_information(state(tdx:end),action(tdx:end),state(tdx-1:end-1),alpha); % I(S;A|S_{t-1})
                I_ss1_a(s,c) = cond_mutual_information(state(tdx:end),state(tdx-1:end-1),action(tdx:end),alpha); % I(S_{t-1};A|S)
                I_as1_s(s,c) = cond_mutual_information(action(tdx:end),state(tdx-1:end-1),state(tdx:end),alpha); % I(S_{t-1};A|S)
                %I_s1a_s(s,c) = cond_mutual_information(state(1:end-1),action(2:end),state(2:end)); % I(S_{t-1};A|S)

                I_s_a(s,c) = mutual_information_basic(state(tdx:end),action(tdx:end),alpha); % I(S;A)
                I_a_s1(s,c) = mutual_information_basic(state(tdx-1:end-1),action(tdx:end)); % I(S_{t-1};A)
                I_s_s1(s,c) = mutual_information_basic(state(tdx:end),state(tdx-1:end-1),alpha); % I(S;S_{t-1})

                % reasonable upper bound on how much info about st-1 is preserved in the agent's observable behavior at time
                I_ssa(s,c) = estimate_ssa(state(tdx-1:end-1), state(tdx:end), action(tdx:end));

                win = 30; % groups of 30 trials
                %for q = 1:length(state)-win

                for q = 1:length(state)-win
                    R_data_mov(s,q,c) = mutual_information_basic(state(q:q+win-1), action(q:q+win-1),alpha);
                    R_data_cond_mi(s,q,c) = cond_mutual_information(state(q+1:q+win-1), action(q+1:q+win-1),state(q:q+win-2),alpha);
                    %synergy_mov(s,q,c) = kernelcmi(state(q+1:q+win-1), action(q+1:q+win-1),state(q:q+win-2)) - kernelmi(state(q:q+win-1), action(q:q+win-1));
                end
            end % cond
        end % subj

        % I_sa_s1 = I_s_a-I_s_s1+I_ss1_a;
        %do people who have a higher mean(I_s_a-I_sa_s1) vs mean(I_a_s1) chunk more?
        sav = I_s_a-I_sa_s1-I_a_s1;
median(I_s_a-I_sa_s1-I_a_s1)
find(sav(:,2)>=-0.1009 & sav(:,4)>=-0.2802)

        figure; hold on;
        nexttile; hold on;
        for c = 1:2
            plot(I_s_a(:,c),reward(:,c),'.','markersize',20,'Color',cmap(c,:))
        end
        ylabel('Average reward')
        xlabel('Policy complexity')
        title('Ns=4'); ylim([0 1]); xlim([0.2 1.5])

        nexttile; hold on;
        for c = 3:4
            plot(I_s_a(:,c),reward(:,c),'.','markersize',20,'Color',cmap(c,:))
        end
        ylabel('Average reward')
        xlabel('Policy complexity')
        title('Ns=6');ylim([0 1]); xlim([0.2 1.5])

        nexttile; hold on;
        b = barwitherr([sem(I_s_a(:,1:2),1)],1, mean(I_s_a(:,1:2))); hold on;
        b(1).FaceColor = cmap(1,:); b(2).FaceColor = cmap(2,:);
        b = barwitherr([sem(I_s_a(:,3:4),1)],2, mean(I_s_a(:,3:4)));
        b(1).FaceColor = cmap(3,:); b(2).FaceColor = cmap(4,:); box off;
        ylabel('I(S;A)'); ylim([0.5 1.2]); title('Policy complexity')
        xticks([1 2]);  xticklabels({'Ns=4', 'Ns=6'});xlabel('Set size')

        nexttile; hold on; colororder(cmap)
        R_data_mov(R_data_mov==0)=NaN;
        plot(squeeze(mean(R_data_mov,1))); ylabel('I(S;A)'); xlabel('Trials');ylim([0.4 1.2])

        nexttile; hold on;
        for c = 1:2
            plot(I_sa_s1(:,c),reward(:,c),'.','markersize',20,'Color',cmap(c,:))
        end
        ylabel('Average reward')
        xlabel('Conditional policy complexity')
        title('Ns=4');  ylim([0 1]); xlim([0.2 1.5])

        nexttile; hold on;
        for c = 3:4
            plot(I_sa_s1(:,c),reward(:,c),'.','markersize',20,'Color',cmap(c,:))
        end
        ylabel('Average reward')
        xlabel('Conditional policy complexity')
        title('Ns=6');ylim([0 1]); xlim([0.2 1.5])

        nexttile; hold on;
        b = barwitherr([sem(I_sa_s1(:,1:2),1)],1, mean(I_sa_s1(:,1:2))); hold on;
        b(1).FaceColor = cmap(1,:); b(2).FaceColor = cmap(2,:);
        b = barwitherr([sem(I_sa_s1(:,3:4),1)],2, mean(I_sa_s1(:,3:4)));
        b(1).FaceColor = cmap(3,:); b(2).FaceColor = cmap(4,:); box off;
        ylabel('I(S;A|S_{t-1})');ylim([0.5 1.2]); title('Conditional policy complexity')
        xticks([1 2]);  xticklabels({'Ns=4', 'Ns=6'})
        xlabel('Set size')

        nexttile; hold on; colororder(cmap)
        R_data_cond_mi(R_data_cond_mi==0)=NaN;
        plot(squeeze(mean(R_data_cond_mi,1))); ylabel('I(S;A|S_{t-1})'); xlabel('Trials');ylim([0.4 1.2])

        set(gcf, 'Position',  [680 730  1300  560])

        % stats
        tbl = table;
        tbl.block_type = categorical(block_type(:));
        tbl.setsize = categorical(setsize(:));
        tbl.complexity = I_s_a(:);
        tbl.cond_complexity = I_sa_s1(:);
        tbl.sub = sub;

        % look at the differences in complexity as a fx of block type and set size
        lme = fitlme(tbl,'complexity ~ block_type*setsize + (block_type|sub) + (setsize|sub)');
        anova(lme)

        % look at the differences in cond_complexity as a fx of block type and set size
        lme = fitlme(tbl,'cond_complexity ~ block_type*setsize + (block_type|sub) + (setsize|sub)');
        anova(lme)

    case 'ICRT_trials'
        figure; hold on;  win = 10; colormap(cmap)
        nexttile; hold on; hold on; ylim([450 1020])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,1)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,2)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Intra-chunk RT (ms)'); title('Ns=4')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'},'Location','SouthEast'); legend('boxoff')

        nexttile; hold on;  ylim([450 1020])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,3)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(ICRT_all_trials(:,:,4)),win,'omitnan'), movmean(sem(ICRT_all_trials(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('Intra-chunk RT (ms)'); title('Ns=6')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'},'Location','SouthEast'); legend('boxoff')

        % ICRT at equilibrium
        nexttile; hold on;  ylim([500 900])
        ICRT_end = [nanmean(ICRT_all_trials(:,15:20,1),2) nanmean(ICRT_all_trials(:,15:20,2),2) nanmean(ICRT_all_trials(:,15:20,3),2) nanmean(ICRT_all_trials(:,15:20,4),2)];
        b = barwitherr([sem(ICRT_end(:,1:2),1)],1,[nanmean(ICRT_end(:,1:2))]); hold on;
        b(1).FaceColor = cmap(1,:); b(2).FaceColor = cmap(2,:);
        b = barwitherr([sem(ICRT_end(:,3:4),1)],2,[nanmean(ICRT_end(:,3:4))]);
        b(1).FaceColor = cmap(3,:); b(2).FaceColor = cmap(4,:);
        ylabel('Intra-chunk RT (ms)');xticks([1:2]); xticklabels({'Ns=4','Ns=6'}); xlabel('Set size')

        % difference in ICRT
        nexttile; hold on; ylim([0 250])
        ICRT_diff = [squeeze(nanmean(ICRT_all_trials(:,15:20,1),2)-nanmean(ICRT_all_trials(:,15:20,2),2)) squeeze(nanmean(ICRT_all_trials(:,15:20,3),2)-nanmean(ICRT_all_trials(:,15:20,4),2))];
        b = barwitherr([sem(ICRT_diff,1)],[nanmean(ICRT_diff)]);
        b(1).FaceColor = [1 1 1]; b(1).LineWidth = 3; b(2).FaceColor = 'k';
        xticks([b(1).XEndPoints b(2).XEndPoints]); xticklabels({'Ns=4','Ns=6'});xlabel('Set size');
        ylabel('\Delta RT (ms)');box off; title('Decrease in Intra-chunk RT')

        % stats
        tbl = table;
        tbl.block_type = categorical(block_type(:));
        tbl.setsize = categorical(setsize(:));
        tbl.icrt = ICRT_end(:);
        tbl.sub = repmat(1:nSubj,1,size(block_type,2))';

        % look at the differences in ICRT as a fx of block type and set size
        lme = fitlme(tbl,'icrt ~ block_type*setsize + (block_type|sub) + (setsize|sub)');
        anova(lme)

        %ttest
        [~,p,~,stat] = ttest(ICRT_diff(:,1),ICRT_diff(:,2));
        disp(['Delta error: t(',num2str(stat.df),') = ',num2str(stat.tstat),', p = ',num2str(p)]);


        win = 30;
        nexttile; hold on; hold on; ylim([450 1020])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,1)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,2)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Non-chunk RT (ms)');

        nexttile; hold on; ylim([450 1020])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,3)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(NCRT_all_trials(:,:,4)),win,'omitnan'), movmean(sem(NCRT_all_trials(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('Non-chunk RT (ms)');

        % NCRT at equilibrium
        nexttile; hold on;  ylim([500 900])
        NCRT_end = [nanmean(NCRT_all_trials(:,50:60,1),2) nanmean(NCRT_all_trials(:,50:60,2),2) nanmean(NCRT_all_trials(:,end-10:end,3),2) nanmean(NCRT_all_trials(:,end-10:end,4),2)];
        b = barwitherr([sem(NCRT_end(:,1:2),1)],1,[nanmean(NCRT_end(:,1:2))]); hold on;
        b(1).FaceColor = cmap(1,:); b(2).FaceColor = cmap(2,:);
        b = barwitherr([sem(NCRT_end(:,3:4),1)],2,[nanmean(NCRT_end(:,3:4))]);
        b(1).FaceColor = cmap(3,:); b(2).FaceColor = cmap(4,:);
        xlabel('Set size');ylabel('Non-chunk RT (ms)');xticks([1:2]); xticklabels({'Ns=4','Ns=6'});

        % difference in NCRT
        nexttile; hold on; ylim([0 250])
        NCRT_diff = [squeeze(nanmean(NCRT_all_trials(:,50:60,1),2)-nanmean(NCRT_all_trials(:,50:60,2),2)) squeeze(nanmean(NCRT_all_trials(:,end-10:end,3),2)-nanmean(NCRT_all_trials(:,end-10:end,4),2))];
        b = barwitherr([sem(NCRT_diff,1)],[nanmean(NCRT_diff)]);
        b(1).FaceColor = [1 1 1]; b(1).LineWidth = 3; b(2).FaceColor = 'k';
        xticks([b(1).XEndPoints b(2).XEndPoints]); xticklabels({'Ns=4','Ns=6'});xlabel('Set size')
        ylabel('\Delta RT (ms)');box off; title('Decrease in Non-chunk RT')

        set(gcf, 'Position',  [400, 100, 1120, 425])

        % NCRT stats
        tbl = table;
        tbl.block_type = categorical(block_type(:));
        tbl.setsize = categorical(setsize(:));
        tbl.ncrt = NCRT_end(:);
        tbl.sub = repmat(1:nSubj,1,size(block_type,2))';

        % look at the differences in NCRT as a fx of block type and set size
        lme = fitlme(tbl,'ncrt ~ block_type*setsize + (block_type|sub) + (setsize|sub)');
        anova(lme)

        %ttest
        [~,p,~,stat] = ttest(NCRT_diff(:,1),NCRT_diff(:,2));
        disp(['Delta error: t(',num2str(stat.df),') = ',num2str(stat.tstat),', p = ',num2str(p)]);

    case 'err_trials'
        figure; hold on;  win = 10; colormap(cmap)
        nexttile; hold on; hold on; ylim([0 1])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,1)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,2)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Intra-chunk error'); title('Ns=4')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'},'Location','NorthEast'); legend('boxoff')

        nexttile; hold on;  ylim([0 1])
        h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,3)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(1-acc_ICRT(:,:,4)),win,'omitnan'), movmean(sem(1-acc_ICRT(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial');  ylabel('Intra-chunk error'); title('Ns=6')
        legend([h(1).mainLine h(2).mainLine],{'Random','Structured'},'Location','NorthEast'); legend('boxoff')

        % error at equilibrium
        nexttile; hold on;  ylim([0 0.25])
        err_end = [nanmean(1-acc_ICRT(:,15:20,1),2) nanmean(1-acc_ICRT(:,15:20,2),2) nanmean(1-acc_ICRT(:,15:20,3),2) nanmean(1-acc_ICRT(:,15:20,4),2)];
        b = barwitherr([sem(err_end(:,1:2),1)],1,[nanmean(err_end(:,1:2))]); hold on;
        b(1).FaceColor = cmap(1,:); b(2).FaceColor = cmap(2,:);
        b = barwitherr([sem(err_end(:,3:4),1)],2,[nanmean(err_end(:,3:4))]);
        b(1).FaceColor = cmap(3,:); b(2).FaceColor = cmap(4,:);
        ylabel('Intra-chunk error');xticks([1:2]); xticklabels({'Ns=4','Ns=6'}); xlabel('Set size')

        % difference in err
        nexttile; hold on; ylim([0 0.2])
        err_diff = [squeeze(nanmean(1-acc_ICRT(:,15:20,1),2)-nanmean(1-acc_ICRT(:,15:20,2),2)) squeeze(nanmean(1-acc_ICRT(:,15:20,3),2)-nanmean(1-acc_ICRT(:,15:20,4),2))];
        b = barwitherr([sem(err_diff,1)],[nanmean(err_diff)]);
        b(1).FaceColor = [1 1 1]; b(1).LineWidth = 3; b(2).FaceColor = 'k';
        xticks([b(1).XEndPoints b(2).XEndPoints]); xticklabels({'Ns=4','Ns=6'}); xlabel('Set size')
        ylabel('\Delta Error');box off; title('Decrease in Intra-Chunk Error')

        % ICRT stats
        tbl = table;
        tbl.block_type = categorical(block_type(:));
        tbl.setsize = categorical(setsize(:));
        tbl.err = err_end(:);
        tbl.sub = repmat(1:nSubj,1,size(block_type,2))';

        % look at the differences in error as a fx of block type and set size
        lme = fitlme(tbl,'err ~ block_type*setsize + (block_type|sub) + (setsize|sub)');
        anova(lme)

        %ttest
        [~,p,~,stat] = ttest(err_diff(:,1),err_diff(:,2));
        disp(['Delta error: t(',num2str(stat.df),') = ',num2str(stat.tstat),', p = ',num2str(p)]);

        % NCRT
        win = 30;
        nexttile; hold on; hold on; ylim([0 1])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,1)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,2)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        xlabel('Trial'); ylabel('Non-chunk error');

        nexttile; hold on; ylim([0 1])
        h(1) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,3)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        h(2) = shadedErrorBar(1:101, movmean(nanmean(1-acc_NCRT(:,:,4)),win,'omitnan'), movmean(sem(1-acc_NCRT(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        xlabel('Trial'); ylabel('Non-chunk error');

        % chunking helps reduce error more in Ns=6 (greater reduction of error in Ns=6
        % than in Ns = 4) showing that people are taking more advantage of chunks

        % error at equilibrium
        nexttile; hold on;  ylim([0 0.25])
        err_end = [nanmean(1-acc_NCRT(:,50:60,1),2) nanmean(1-acc_NCRT(:,50:60,2),2) nanmean(1-acc_NCRT(:,end-10:end,3),2) nanmean(1-acc_NCRT(:,end-10:end,4),2)];
        b = barwitherr([sem(err_end(:,1:2),1)],1,[nanmean(err_end(:,1:2))]); hold on;
        b(1).FaceColor = cmap(1,:); b(2).FaceColor = cmap(2,:);
        b = barwitherr([sem(err_end(:,3:4),1)],2,[nanmean(err_end(:,3:4))]);
        b(1).FaceColor = cmap(3,:); b(2).FaceColor = cmap(4,:);
        ylabel(' Non-chunk error');xticks([1:2]); xticklabels({'Ns=4','Ns=6'}); xlabel('Set size')

        % difference in err
        nexttile; hold on; ylim([0 0.2])
        err_diff = [squeeze(nanmean(1-acc_NCRT(:,50:60,1),2)-nanmean(1-acc_NCRT(:,50:60,2),2)) squeeze(nanmean(1-acc_NCRT(:,end-10:end,3),2)-nanmean(1-acc_NCRT(:,end-10:end,4),2))];
        b = barwitherr([sem(err_diff,1)],[nanmean(err_diff)]);
        b(1).FaceColor = [1 1 1]; b(1).LineWidth = 3; b(2).FaceColor = 'k';
        xticks([b(1).XEndPoints b(2).XEndPoints]); xticklabels({'Ns=4','Ns=6'}); xlabel('Set size')
        ylabel('\Delta Error');box off; title('Decrease in Non-Chunk Error')

        set(gcf, 'Position',  [400, 100, 1120, 425])

        % stats
        tbl = table;
        tbl.block_type = categorical(block_type(:));
        tbl.setsize = categorical(setsize(:));
        tbl.err = err_end(:);
        tbl.sub = repmat(1:nSubj,1,size(block_type,2))';

        % look at the differences in error as a fx of block type and set size
        lme = fitlme(tbl,'err ~ block_type*setsize + (block_type|sub) + (setsize|sub)');
        anova(lme)

        %ttest
        [~,p,~,stat] = ttest(err_diff(:,1),err_diff(:,2));
        disp(['Delta error: t(',num2str(stat.df),') = ',num2str(stat.tstat),', p = ',num2str(p)]);

        %         figure; hold on;  win = 10; colormap(cmap)
        %         nexttile; hold on; hold on; ylim([0 1])
        %         h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,1)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,1),1),win,'omitnan'),{'color',cmap(1,:)},1);
        %         h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,2)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,2),1),win,'omitnan'),{'color',cmap(2,:)},1);
        %         xlabel('Trial'); ylabel('Intra-chunk accuracy'); title('Ns=4')
        %         legend([h(1).mainLine h(2).mainLine],{'Random','Structured'}); legend('boxoff')
        %
        %         nexttile; hold on;  ylim([0 1])
        %         h(1) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,3)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        %         h(2) = shadedErrorBar(1:nTrials, movmean(nanmean(acc_ICRT(:,:,4)),win,'omitnan'), movmean(sem(acc_ICRT(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
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
        %         h(1) = shadedErrorBar(1:101, movmean(nanmean(acc_NCRT(:,:,3)),win,'omitnan'), movmean(sem(acc_NCRT(:,:,3),1),win,'omitnan'),{'color',cmap(3,:)},1);
        %         h(2) = shadedErrorBar(1:101, movmean(nanmean(acc_NCRT(:,:,4)),win,'omitnan'), movmean(sem(acc_NCRT(:,:,4),1),win,'omitnan'),{'color',cmap(4,:)},1);
        %         xlabel('Trial'); ylabel('Non-chunk accuracy');

end


end