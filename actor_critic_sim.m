function simdata = actor_critic_sim(agent, data)

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

    % theta = reward;
    rho = 0;
    ecost = 0;
    inChunk = 0;
    chunkStep = 0; stop = 0;
    g = [];
    p = ones(1,nA); p = p./sum(p); % transition probability matrix
     rho0 = 0;ecost0 = 0;
    for t = 1:length(state)
        s = state(t);

        % full RL policy
        d = beta*theta(s,:)+log(p);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);    % softmax policy
        entropy = -nansum(policy.*log2(policy));

        a = fastrandsample(policy); % action
        action(t) = a;
        r = reward(s, a);           % reward
        cost = logpolicy(a) - log(p(a));   % policy complexity cost

        log_rt = log(agent.t0 + agent.b1*abs(cost) + agent.b2*entropy) + normrnd(0,agent.sigma^2);
        rt = exp(log_rt);

        rpe = beta*r - cost - V(s);
        rho = rho + agent.lrate_r*(r-rho);             % avg reward update
        ecost = ecost + agent.lrate_e*(cost-ecost);    % policy cost update

        chosen = a; idxs = 1:nA; unchosen = idxs(idxs~=chosen);
        g(:,chosen) = beta*(1-policy(chosen));       % policy gradient for chosen actions
        g(:,unchosen) = beta*(-policy(unchosen));    % policy gradient for unchosen actions
        theta(s,:) = theta(s,:) + agent.lrate_theta*rpe*g;  % policy parameter update
        V(s) = V(s) + agent.lrate_V*rpe;

        if agent.lrate_beta > 0 %&& agent.C < ecost
            beta = beta + agent.lrate_beta*(agent.C-ecost);
            %beta = beta + agent.lrate_beta*(rho0-rho);
        %elseif agent.lrate_beta > 0 && rho > agent.V
            %beta = beta + agent.lrate_beta*(ecost0-ecost);
            %beta = beta + agent.lrate_beta*(rho0-rho);
            %beta = beta + agent.lrate_beta*(agent.C-ecost);
        end
        %
        %         R = 0.85;
        %         if agent.lrate_beta > 0 && rho0 < R
        %             %R = 0.85;
        %             beta = beta + agent.lrate_beta*(rho0-rho);
        %             %beta = beta + agent.lrate_beta*(R-rho);
        %         end
        beta = max(0,min(beta,10));

        if agent.lrate_p > 0 && t>1
            p= p + agent.lrate_p*(policy - p);
            p = p./sum(p);  % marginal update
        end
        rho0 = rho;

        simdata.s(idx(t),:) = s;
        simdata.a(idx(t),:) = a;
        simdata.r(idx(t),:) = r;
        simdata.rt(idx(t),:) = rt;
        simdata.acc(idx(t),:) = s==a;
        simdata.beta(idx(t),:) = beta;
        simdata.ecost(idx(t),:) = ecost;
        simdata.cost(idx(t),:) = cost;
        simdata.cond(idx(t),:) = condition(t);
        simdata.theta{idx(t)} = theta;
        simdata.inChunk(idx(t),:) = inChunk;
        simdata.chunkStep(idx(t),:) = chunkStep;
        simdata.policy{idx(t)} = policy;
        %if t > 1 && strcmp(condition(t),condition(t-1))==0
        %    ecost
        %end
    end
    simdata.p(block) = {p};
end % condition

end % chunk

