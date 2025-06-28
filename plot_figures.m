function plot_figures(fig, data)


if nargin <2
    load actionChunk_data.mat;              % original dataset
    %load actionChunk_revision_data.mat;     % changed for consistent # of actions across both set sizes
    %load actionChunk_prob_data.mat;         % probabalistic random test (50% of trials switch to random test)
    %load actionChunk_timepressure_data.mat; % shorted time window of test response to 1000ms and train response to 2000ms and make structured to random test continuous
    %load data2.mat                            % original but with warning
end

%load models_RT.mat;
%model = 4; simdata = sim_fitted(model, data, results);
%model = 4; simdata_chunk = sim_fitted(model, data, results);
%simdata_chunk(1).chunk = 1; simdata_chunk(1).sim = 'sim_'; simdata_chunk(1).revis = [];
%[lme, simdata_chunk] = lme_RT('rt ~ -1 + complexity + entropy  + (-1+complexity+entropy|sub)',model,simdata_chunk);
%model = 3; simdata_nochunk = sim_fitted(model, data, results); simdata_chunk(1).chunk = 0;
%[lme, simdata_nochunk] = lme_RT('rt ~ -1 + complexity + entropy  + (-1+complexity+entropy|sub)',model); simdata_nochunk(1).revis = [];


conds = {'Ns4,random_train', 'Ns4,structured_train', 'Ns4,structured_test','Ns4,random_test',...
    'Ns6,random_train', 'Ns6,structured_train', 'Ns6,structured_test','Ns6,random_test'};
nSubj = length(data);
plot_log = 0; % plot RT in log space?
% analysis in log space?
if plot_log ==1
    for s = 1:nSubj
        data(s).rt = log(data(s).rt);
    end
end

switch fig
    case 'revision'
        %load data
        simdata = sim_manual(data);
        learning_curve(simdata);
        RT_curve(simdata);
        %analysis_data('ICRT_correct_chunk',simdata);
        analysis = analysis_data('avgAcc', simdata);
        %analysis_data('avgRT', simdata);
        %analysis_data('ICRT_correct_chunk',simdata); sgtitle('Correct')
        %analysis_data('ICRT_trials_Ns4', simdata)
        %analysis_data('ICRT_trials_Ns6', simdata)
        %analysis_data('ICRT_vs_NCRT', simdata)
        %analysis_data('corr_RT_train_test', simdata)
        %analysis_data('actionSlips', simdata)
        data = simdata;

        %entropy = nan(nSubj, 120, length(conds));
        chunkInit = [2,5];
        for s = 1:nSubj
            beta(s,:) = data(s).beta;
            b = []; complex = [];
            for c = 1:length(conds)
                if contains(conds(c),'4')
                    condIdx = 1;
                elseif contains(conds(c), '6')
                    condIdx = 2;
                end
                idx = strcmp(data(s).cond, conds(c));
                state = data(s).s(idx);
                action = data(s).a(idx);
                acc = data(s).s(idx)==data(s).a(idx);
                rt = data(s).rt(idx);
                r = data(s).r(idx);
                beta_bar(s,c) = mean(data(s).beta(idx));
                b = [b; data(s).beta(idx)];
                complex = [complex; data(s).ecost(idx)];
                reward(s,c) = mean(r);
                %complexity(s,c) = mean(data(s).ecost(idx));
                I_a_s(s,c) = mean(data(s).cost(idx));
                I_a_s(s,c) = mutual_information_basic(state,action,0.1);
                I_a_s1(s,c) = mutual_information_basic(state(1:end-1),action(2:end),0.1);
                %entropy(s,1:length(data(s).entropy(idx)),c) = data(s).entropy(idx);
            end
            B(s,:) = b;
            C(s,:) = complex;
        end

        figure;hold on;
        nexttile; hold on;
        plot(mean(B,1)); xlabel('Trials'); ylabel('\beta')
        xline([80   160   220   280   400   520   610   700])
        nexttile; hold on;
        plot(mean(C,1)); xlabel('Trials'); ylabel('Policy complexity');
        ylim([0 0.7])
        xline([80   160   220   280   400   520   610   700])
        yline([2])

        cmap =[141 182 205
            255 140 105
            238 201 0
            155 205 155] / 255;

        figure; hold on;
        colororder(cmap);
        avgComplex = mean(I_a_s);
        avgComplex = reshape(avgComplex,4,2)';

        b = bar(avgComplex); se = sem(I_a_s,1); se = reshape(se,4,2)';
        errorbar_pos = errorbarPosition(b, se);
        errorbar(errorbar_pos', avgComplex, se, se, 'k','linestyle','none', 'lineWidth', 1.2,'capsize',0);
        set(gca, 'XTick',1:2, 'XTickLabel', {'Ns=4', 'Ns=6'});
        ylabel('Policy Complexity');
        legend('Random Train', 'Structured Train','Structured Test', 'Random Test','Location', 'northwest'); legend('boxoff');
        ylim([0 0.7]);

        set(gcf, 'Position',  [400, 400, 650, 300])

        figure; hold on;
        bar(1,[mean(analysis.acc(:,2)-analysis.acc(:,1)) mean(analysis.acc(:,3)-analysis.acc(:,4))])
        bar(2,[mean(analysis.acc(:,6)-analysis.acc(:,5)) mean(analysis.acc(:,7)-analysis.acc(:,8))])
        ylabel('\Delta Structured-Random Accuracy')
        xticks([1:2]);xticklabels({'Ns4','Ns6'});ylim([0 0.3])

        figure(100); hold on;
        bar(1,[mean(I_a_s(:,1)) mean(I_a_s(:,5))])
        %xticks([1:2]);xticklabels({'Ns4','Ns6'});ylim([0 0.3])

        % main point should be
    case 'plot_all_data' % All data plots

        for s = 1:nSubj
            C = [];
            for c = 1:length(conds)
                idx = strcmp(data(s).cond, conds(c));
                state = data(s).s(idx);
                action = data(s).a(idx);
                accuracy(s,c) = sum(state==action);
                for ss = 1:max(state)
                    acc(s,ss,c) = sum(state==ss & action==ss)./sum(state==ss);
                end
                reward(s,c) = mean(data(s).r(idx));
                mi(s,c) = kernelmi(state,action);
                cond_mi(s,c) = kernelcmi(state(2:end),action(2:end),state(1:end-1));
                I_s_s1(s,c) = kernelmi(state(1:end-1),state(2:end));
                I_s_s1_a(s,c) = kernelcmi(state(1:end-1),state(2:end),action(2:end));
                I_a_s(s,c) = mutual_information_basic(state,action,0.1);
                I_a_s1(s,c) = mutual_information_basic(state(1:end-1),action(2:end),0.1);


                win = 20; % groups of 30 trials
                for q = 1:length(state)-win
                    R_data_mov(s,q,c) = mutual_information(state(q:q+win-1), action(q:q+win-1),0.1);
                    R_data_mov2(s,q,c) = mutual_information(state(q:q+win-2), action(q+1:q+win-1),0.1);
                    R_data_cond_mi(s,q,c) = kernelcmi(state(q+1:q+win-1), action(q+1:q+win-1),state(q:q+win-2));
                    synergy_mov(s,q,c) = kernelcmi(state(q+1:q+win-1), action(q+1:q+win-1),state(q:q+win-2)) - kernelmi(state(q:q+win-1), action(q:q+win-1));
                  R_data_cond_mi2(s,q,c) = kernelcmi(action(q+1:q+win-1),state(q:q+win-2),state(q+1:q+win-1));
                    
                    %V_data_mov(s,q,c) = mean(reward(q:q+win-1));
                    %rt_mov(s,q,c) = mean(RT(q:q+win-1));
                end

            end
        end
        interaction = mi - cond_mi;
        
        cmap =[141 182 205
            255 140 105
            238 201 0
            155 205 155] / 255;

        R_data_mov(R_data_mov==0)=NaN;
        figure; hold on; colororder(cmap)
        plot(squeeze(mean(R_data_mov,1))); ylabel('I(S;A)'); xlabel('Trials');ylim([0.3 1.2])


        R_data_mov2(R_data_mov2==0)=NaN;
        figure; hold on; colororder(cmap)
        plot(squeeze(mean(R_data_mov2,1))); ylabel('I(S_{t-1};A)'); xlabel('Trials');ylim([0.3 1.2])

        R_data_cond_mi(R_data_cond_mi==0)=NaN;
        figure; hold on; colororder(cmap)
        plot(squeeze(mean(R_data_cond_mi,1))); ylabel('I(S;A|S_{t-1})'); xlabel('Trials');ylim([0.3 1.2])
 
        synergy_mov(synergy_mov==0)=NaN;
        figure; hold on; colororder(cmap)
        plot(squeeze(mean(synergy_mov,1))); ylabel('Synergy'); xlabel('Trials');


        R_data_cond_mi2(R_data_cond_mi2==0)=NaN;
        figure; hold on; colororder(cmap)
        plot(squeeze(mean(R_data_cond_mi2,1))); ylabel('I(S_{t-1};A|S)'); xlabel('Trials');ylim([0.3 1.2])
 

        figure; hold on; colororder(cmap)
        nexttile; hold on;bar(1,mean(mi)); ylabel('I(S;A)'); ylim([0 1.2])
        nexttile; hold on;bar(1,mean(I_s_s1)); ylabel('I(S;S_{t-1})');ylim([0 1.2])
        nexttile; hold on;bar(1,mean(I_s_s1_a)); ylabel('I(S;S_{t-1}|A)');ylim([0 1.2])
        nexttile; hold on;bar(1,mean(I_a_s1)); ylabel('I(S_{t-1};A)');ylim([0 1.2])
        nexttile; hold on;bar(1,mean(cond_mi)); ylabel('I(S;A|S_{t-1})');ylim([0 1.2])
        nexttile; hold on;bar(1,mean(interaction)); ylabel('I(S;A;S_{t-1})');ylim([-0.5 0.5])

        nexttile; hold on;bar(1,mean(cond_mi-mi)); ylabel('I(S;A|S_{t-1})-I(S;A)');ylim([-0.2 0.5])
        %load data

        figure; hold on; colororder(cmap)
         nexttile; hold on;
         for c = 1:4
            plot(cond_mi(:,c),reward(:,c),'.','markersize',20)
        end
        ylabel('Average Reward')
        xlabel('Policy Complexity')
        nexttile; hold on;
        for c = 5:8
            plot(cond_mi(:,c),reward(:,c),'.','markersize',20)
        end
        ylabel('Average Reward')
        xlabel('Policy Complexity')


        learning_curve(data);
        RT_curve(data);
        analysis_data('avgAcc', data);
        %analysis_data('avgRT', data);
        %analysis_data('ICRT_all_train', data)
        analysis_data('ICRT_correct_chunk',data); sgtitle('Correct')
        %analysis_data('actionSlips', data)
        %analysis_data('corr_RT_train_test', data)

        %analysis_data('ICRT_incorrect_chunk',data); sgtitle('Incorrect')
        %pause(1)
        %analysis_data('corr_RT_train_test', data)
        %analysis_data('ICRT_trials_Ns4', data)
        %analysis_data('ICRT_trials_Ns6', data)
        %analysis_data('ICRT_vs_NCRT', data)
        %analysis_data('actionSlips', data)
        %analysis_policy_complexity('avgComplexity',data,0); sgtitle('Not Recoded')
        %analysis_policy_complexity('avgComplexity',data,1); sgtitle('Recoded')


        figure; hold on; bar(mean(beta_bar))
        %figure; hold on; colororder(cmap)
        %plot(squeeze(mean(entropy,1)))

    case 'model'

        analysis_policy_complexity('avgComplexity',data,0) %
        analysis_policy_complexity('RC_curves',data,0) %


        chunkInit = [2,5];
        for s = 1:nSubj
            beta(s,:) = data(s).beta;
            for c = 1:length(conds)
                if contains(conds(c),'4')
                    condIdx = 1;
                elseif contains(conds(c), '6')
                    condIdx = 2;
                end
                idx = strcmp(data(s).cond, conds(c));
                state = data(s).s(idx);
                action = data(s).a(idx);
                rt = data(s).rt(idx);
                r = data(s).r(idx);
                beta_bar(s,c) = mean(data(s).beta(idx));
                reward(s,c) = mean(r);
                %complexity(s,c) = mean(data(s).ecost(idx));
                I_a_s(s,c) = mean(data(s).cost(idx));
            end
        end

        figure; hold on;
        bar(mean(beta_bar)); ylabel('beta')


        figure; hold on;
        plot_RPCcurve(reward, I_a_s, [1 2], {'Ns=4, Random Train', 'Ns=4, Structured Train'},'setsize',1);
        plot_RPCcurve(reward, I_a_s, [5 6], {'Ns=6, Random Train', 'Ns=6, Structured Train'},'setsize',1);
        plot_RPCcurve(reward, I_a_s, [3 4], {'Ns=4, Structured Test', 'Ns=4, Random Test'},'setsize',1);
        plot_RPCcurve(reward, I_a_s, [7 8], {'Ns=6, Structured Test', 'Ns=6, Random Test'},'setsize',1);

        set(gcf, 'Position',  [400, 400, 550, 500])


    case 'fig3' % Model recovery analysis

    case 'fig4' % Participant performance on task (data)
        learning_curve(data);
        RT_curve(data);
        analysis_data('avgAcc_train', data);
        analysis_data('avgRT_train', data);

    case 'fig5' % Reaction time as a product of policy entropy and complexity cost (data & model)
        analysis_model('RT_toymodel')
        [lme, simdata_chunk] = lme_RT('rt ~ -1 + complexity + entropy  + (-1+complexity+entropy|sub)',model);
        [lme, simdata_entropy] = lme_RT('rt ~ -1 + entropy + (-1+entropy|sub)',model); % entropy only
        [lme, simdata_complexity] = lme_RT('rt ~ -1 + complexity + (-1+complexity|sub)',model);  % complexity only

    case 'fig6'  % ICRT (data & chunk model & noChunk model)
        analysis_data('ICRT_all_train', data)
        analysis_data('ICRT_all_train',simdata_chunk)
        analysis_data('ICRT_all_train',simdata_entropy)
        analysis_data('ICRT_all_train',simdata_complexity)

    case 'fig7' % Chunkers' ICRT behavior is better accounted for by the chunk model and NonChunkers' ICRT behavior is better accounted for by the noChunk model (data & model)
        analysis_data('ICRT_correct_chunk',data)
        analysis_data('ICRT_correct_chunk', simdata_chunk)
        analysis_data('ICRT_correct_chunk', simdata_nochunk)
        % TODO: clean this up

    case 'fig8'  % Chunk use predicts RTs after chunk-breaking (data & model)
        analysis_data('corr_RT_train_test', data)
        analysis_data('corr_RT_train_test', simdata_chunk)

    case 'fig9'
        analysis_data('ICRT_complexity', simdata_chunk)
        analysis_data('ICRT_complexity', simdata_nochunk)

    case 'fig10'
        specs.all = 0; specs.chunkers = 1; specs.recode = 1;
        analysis_policy_complexity('RC_curves', specs)
        specs.all = 0; specs.chunkers = 0; specs.recode = 0;
        analysis_policy_complexity('RC_curves', specs)

    case 'fig11'
        specs.all = 0; specs.chunkers = 1; specs.recode = 1;
        analysis_policy_complexity('RC_bars', specs)
        specs.all = 0; specs.chunkers = 0; specs.recode = 0;
        analysis_policy_complexity('RC_bars', specs)

    case 'fig12'
        specs.all = 0; specs.chunkers = 1; specs.recode = 1;
        analysis_policy_complexity('RC_curves', specs, simdata_chunk)
        specs.all = 0; specs.chunkers = 0; specs.recode = 0;
        analysis_policy_complexity('RC_curves', specs, simdata_nochunk)

    case 'fig14'
        specs.all = 0; specs.chunkers = 1; specs.recode = 1;
        analysis_policy_complexity('RC_bars', specs, simdata_chunk)
        specs.all = 0; specs.chunkers = 0; specs.recode = 0;
        analysis_policy_complexity('RC_bars', specs, simdata_nochunk)


    case 'supplemental'
        plotBIC()

    case 'extra' % People and cost-sensitive agents chunk more under a higher working memory load (data & model)
        analysis_data('actionSlips', simdata_chunk)
        analysis_model('inchunk', simdata_chunk)
        analysis_data('ICRT_trials_Ns4', data)
        analysis_data('ICRT_trials_Ns6', data)
        analysis_data('ICRT_trials_Ns4', simdata_chunk)
        analysis_data('ICRT_trials_Ns6', simdata_chunk)
        analysis_policy_complexity('RC_curves')
        analysis_policy_complexity('RC_bars')

        % Polynomial regression modeling of empirical reward-complexity curves
        analysis_policy_complexity('RC_empirical_curves')

        % Chunking reduces policy complexity (model)
        specs.all = 0; specs.chunkers = 1; specs.recode = 1;
        analysis_policy_complexity('RC_bars', simdata_chunk, specs);
        analysis_policy_complexity('RC_curves', simdata_chunk, specs)

end


end