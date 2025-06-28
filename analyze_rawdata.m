function data = analyze_rawdata(experiment)

%{
    Analyze raw jsPsych experiment data saved in .csv files

    USAGE:
        data = analyze_rawdata('setsize')
        data = analyze_rawdata('setsize_revision')
        data = analyze_rawdata('setsize_prob_revision')
%}

prettyplot;

switch experiment
    case 'bonus'
        folder = 'experiment_tp_warning_20240108/data/';
        subj = {'A3IMM3YOEK9D3H','A2O6ZD4I5G8N6S','A1ATQ2AVZB2ND5','A2PUL3ZDXOW0VZ','A2OO4PG3LBLP5I'};

        for s = 1:length(subj)
            A = readtable(strcat(folder, subj{s}));
            A = table2cell(A);
            corr = sum(strcmp(A(3:end, 13), 'true'));
            incorr = sum(strcmp(A(3:end,13), 'false'));
            data(s).performance = corr/(corr+incorr);
            data(s).bonus = round(mean(data(s).performance)*7, 2);
        end
        bonus = [cellstr(subj') {data.bonus}'];
        writecell(bonus,'bonus.csv');

    case 'setsize'
        folder = 'experiment_tp_warning_20240108/data/';
        %subj1 = {'A2O6ZD4I5G8N6S','A2O6ZD4I5G8N6S','A2O6ZD4I5G8N6S','A1ATQ2AVZB2ND5','A2PUL3ZDXOW0VZ','A2OO4PG3LBLP5I','A1ATQ2AVZB2ND5','A2PUL3ZDXOW0VZ','A2OO4PG3LBLP5I','A1ATQ2AVZB2ND5','A2PUL3ZDXOW0VZ','A2OO4PG3LBLP5I','A1ATQ2AVZB2ND5','A2PUL3ZDXOW0VZ','A2OO4PG3LBLP5I'};
        subj1 = {'A3IMM3YOEK9D3H','A2O6ZD4I5G8N6S','A1ATQ2AVZB2ND5','A2PUL3ZDXOW0VZ','A2OO4PG3LBLP5I','AJDXSXAWDDAEO','A3CCFD0700KTPV','A3Q6YRWUMALT6D','A3I40B0FATY8VH',...
            'A2U50SMRZT60ZF','ASVRLMDNQBUD9','A11FRLH5KWRLBV','A3DW6KSQPG6GVQ',...
            'AUZNL6ARA1UEC','A3FES5R5CDC91M','A1OZPLHNIU1519','A10249252O9I20MRSOBVF',...
            'A2NXMRPHG86N2T','A3RLCGRXA34GC0','A36SM7QM8OK3H6','A2QWR7XEBV7ADK',...
            'A2ML0070M8FDK1','AIXZNUNYZHM4I','A23Z8G9ARHNM96','A24MNMHSFYW6B',...
            'A2DVV59R1CQU6T','A25FH7PXC446RG','AQ9Y6WD8O72ZC','ACJ6NSCIWMUZI'};

        subj = [subj1];
        nTrials = 700;
        savepath = 'data2.mat';

        startOfExp = 4;  %change
        %data.cutoff = 0.56;
        pcorr = zeros(length(subj),1);

        for s = 1:length(subj)
            % 1.rt  2.url  3.trial_type  4.trial_index  5.time_elapsed  % 6.internal_node_id
            % 7.view_history  8.stimulus  9.key_pressed  10.state  11.test_part
            % 12.correct_response  13.correct  14.bonus  15.responses

            A = readtable(strcat(folder, subj{s}));
            A = table2cell(A);

            corr = sum(strcmp(A(startOfExp:end, 13), 'true'));
            incorr = sum(strcmp(A(startOfExp:end,13), 'false'));
            pcorr(s) = corr/(corr+incorr);
        end

        figure; hold on;
        histogram(pcorr, 20, 'FaceColor', '#0072BD');
        xlabel('% Accuracy'); ylabel('# of Subjects');
        box off; set(gcf,'Position',[200 200 800 300]);
        %subj = subj(pcorr>cutoff); % filter by correct probability > cutoff

        % Construct data structure
        for s = 1:length(subj)
            A = readtable(strcat(folder, subj{s}));
            A = table2cell(A);
            corr = sum(strcmp(A(startOfExp:end, 13), 'true'));
            incorr = sum(strcmp(A(startOfExp:end,13), 'false'));
            data(s).performance = corr/(corr+incorr);
            data(s).bonus = round(data(s).performance * 8, 2);

            A(:,13) = strrep(A(:,13), 'true', '1');
            A(:,13) = strrep(A(:,13), 'false', '0');

            condition = unique(A(:,11));
            condition(strcmp(condition, '')) = [];
            expTrialIdx = ismember(A(:,11), condition);
            A(strcmp(A, 'null'),9) = {'-1'};
            data(s).ID = subj{s};

            data(s).cond = A(expTrialIdx, 11);
            data(s).idx = A(expTrialIdx, 4);
            data(s).s = cell2mat(A(expTrialIdx, 10));
            data(s).a = str2num(cell2mat(A(expTrialIdx, 9))) - 48;
            data(s).a(data(s).a==-49) = NaN;
            data(s).corrchoice = cell2mat(A(expTrialIdx,12)) - 48;
            data(s).acc = str2num(cell2mat(A(expTrialIdx, 13)));
            data(s).r = str2num(cell2mat(A(expTrialIdx, 13)));
            data(s).rt = zeros(nTrials, 1);
            idx = find(expTrialIdx);
            for i = 1:sum(expTrialIdx)
                data(s).rt(i) = str2double(A{idx(i),1});
            end
            data(s).N = nTrials;

            conds = unique(data(s).cond);
            for i = 1:length(conds)
                nans(i) = sum(isnan(data(s).a(strcmp(conds(i),data(s).cond))));
                ntrials(i) = length(data(s).a(strcmp(conds(i),data(s).cond)));
            end
            if any(nans./ntrials >.3)
                flag(s) = 1;
                n(s,:) = nans./ntrials;
            end
        end % for each subject

        data = data(flag==0);
        data(19:32) = data(1:14);
        data(1).revis = [];
        data(1).sim = [];
        save(savepath, 'data');

end