function simdata = actor_critic_sim_chunk(agent, data)

%{
    Simulation of the action sequences for both the set size manipulation experiment
    and the incentive & load manipulation experiment.
    
%}
if ~isfield(agent, 'beta')
    agent.beta = agent.beta0;
end

%cond = {{'Ns4,random_train'}, {'Ns4,structured_train', 'Ns4,structured_test','Ns4,random_test'},...
%    {'Ns6,random_train'}, {'Ns6,structured_train', 'Ns6,structured_test','Ns6,random_test'}};

cond = {{'Ns4,random_train'}, {'Ns4,structured_train'}, {'Ns6,random_train'}, {'Ns6,structured_train'}};

for block = 1:length(cond)
    idx = [];
    expCond = cond{block};
    for c = 1:length(expCond)
        idx = [idx find(strcmp(data.cond, expCond{c}))'];
    end
    condition = data.cond(idx);
    state = data.s(idx);
    action = data.a(idx);
    rewards = data.r(idx);
    corrchoice = state;
    setsize = length(unique(state));
    nA = max(unique(action));
    theta = zeros(setsize, nA);
    V = zeros(setsize,1);
    if contains(agent.m,'fixed')
        beta = agent.beta(block);
    else
        beta = agent.beta;
    end

    if setsize==4
        chunk = [2 1];
        if ~exist('incentives', 'var')
            if nA > 4
                reward = [eye(4), transpose([0 0 0 0]), transpose([0 0 0 0])];
            else
                reward = eye(4);
            end
        else
            reward = incentives.Ns4;
        end

    end
    if setsize==6
        chunk = [5 4];
        if ~exist('incentives', 'var')
            reward = eye(6);
        else
            reward = incentives.Ns6;
        end
    end

    %reward(reward==0) = -1;
    %theta = reward;
    rho = 0;
    ecost = 0; ecost_cond = 0;
    inChunk = 0;
    chunkStep = 0;stop = 0;
    g = [];
    p = ones(setsize,nA); p = p./sum(p(1,:)); % transition probability matrix
    pp = ones(1,nA); pp = pp./sum(pp); % transition probability matrix
    ps = ones(setsize,setsize); ps = ps./sum(ps(1,:)); % transition probability matrix
    rho0 = 0;ecost0 = 0;


    if contains(expCond{1},'Ns4,structured')
        p = [1 3 3 3;
            3 1 1 1;
            1 3 3 3;
            1 3 3 3];

        p = [1 2 2 2;
            2 1 1 1;
            1 2 2 2;
            1 2 2 2];
    elseif contains(expCond{1},'Ns6,structured')
        p = [3 3 3 1 3 3;
            3 3 3 1 3 3;
            3 3 3 1 3 3;
            3 3 3 1 3 3;
            1 1 1 3 1 1;
            3 3 3 1 3 3];

        p = [2 2 2 1 2 2;
            2 2 2 1 2 2;
            2 2 2 1 2 2;
            2 2 2 1 2 2;
            1 1 1 2 1 1;
            2 2 2 1 2 2];
    end
    p = p./sum(p,2);
    for t = 1:length(state)

        s = state(t);
        if t > 1
            prior = p(state(t-1),:);
        else
            prior = p(1,:);
        end

        % full RL policy
        d = beta*theta(s,:)+log(prior);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);    % softmax policy
        entropy = -nansum(policy.*log2(policy));

        if t>1
            po = policy.*ps(state(t-1),:);
            pr = prior.*ps(state(t-1),:);
        end
        a = fastrandsample(policy); % action
        action(t) = a;
        r = reward(s, a);           % reward
        cost = logpolicy(a) - log(prior(a));   % policy complexity cost
        %cost = sum(policy.*logpolicy-log(prior));
        cost_cond = logpolicy(a) - log(pp(a));   % policy complexity cost

        log_rt = log(agent.t0 + agent.b1*abs(cost) + agent.b2*entropy) + normrnd(0,agent.sigma^2);
        rt = exp(log_rt);
        rpe = beta*r - cost - V(s);
        rho = rho + agent.lrate_r*((r==1)-rho);        % avg reward update
        ecost = ecost + agent.lrate_e*(cost-ecost);    % policy cost update
        ecost_cond = ecost_cond + agent.lrate_e*(cost_cond-ecost_cond);    % policy cost update

        chosen = a; idxs = 1:nA; unchosen = idxs(idxs~=chosen);
        g(:,chosen) = beta*(1-policy(chosen));       % policy gradient for chosen actions
        g(:,unchosen) = beta*(-policy(unchosen));    % policy gradient for unchosen actions
        theta(s,:) = theta(s,:) + agent.lrate_theta*rpe*g;  % policy parameter update
        V(s) = V(s) + agent.lrate_V*rpe;

        if agent.lrate_beta > 0 %&& agent.C<ecost
            beta = beta + agent.lrate_beta*(agent.C-ecost);
        end
        ecost = max(ecost,0);

        if t==20
            %  keyboard
        end

        if agent.lrate_p > 0 && t>1
            p(state(t-1),:) = p(state(t-1),:) + agent.lrate_p*(policy - p(state(t-1),:));
            p = p./sum(p,2);  % marginal update
        end
        if agent.lrate_p > 0 && t>1
            pp = pp + agent.lrate_p*(policy - pp);
            pp = pp./sum(pp,2);  % marginal update
        end

        if agent.lrate_p > 0 && t>1
            v = zeros(1,max(state)); v(state(t)) = 1;
            ps(state(t-1),:) =  ps(state(t-1),:) + agent.lrate_p*(v-ps(state(t-1),:));
            ps = ps./sum(ps,2);  % marginal update
        end
        rho0 = rho;
        ecost0 = ecost;
        simdata.s(idx(t),:) = s;
        simdata.a(idx(t),:) = a;
        simdata.r(idx(t),:) = r;
        simdata.rt(idx(t),:) = rt;
        simdata.acc(idx(t),:) = s==a;
        simdata.beta(idx(t),:) = beta;

        simdata.rho(idx(t),:) = rho;
        simdata.ecost(idx(t),:) = ecost;
        simdata.ecost_cond(idx(t),:) = ecost_cond;
        simdata.cost(idx(t),:) = cost;
        simdata.cond(idx(t),:) = condition(t);
        simdata.theta{idx(t)} = theta;
        simdata.inChunk(idx(t),:) = inChunk;
        simdata.chunkStep(idx(t),:) = chunkStep;
        simdata.policy{idx(t)} = policy;
    end
    simdata.p(block) = {p};
end % condition

end % chunk

