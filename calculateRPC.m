function [reward, complexity] = calculateRPC(data, conds, recode, maxReward, chunk_only, split_idx)
if nargin<6;split_idx = 1;end
nSubj = length(data);
if recode
    complexity = recoded_policy_complexity(data, conds, split_idx);
end

chunkInit = [2 5];
reward = nan(nSubj, length(conds));
for s = 1:nSubj
    for c = 1:length(conds)
        idx = strcmp(data(s).cond, conds(c));
        state = data(s).s(idx);
        action = data(s).a(idx);
        reward(s,c) = sum(data(s).r(idx));
        
        if split_idx > 1
            state = state(1:split_idx);
            action = action(1:split_idx);
        end
        nS = length(unique(state));
        if ~recode
            complexity(s,c) = mutual_information_basic(state,action,0.1); 
        end
        %if ~recode; complexity(s,c) = mutual_information(state,action,0.01, repmat(1/nS,1,nS)); end
    end
end
reward = reward ./ maxReward;

end