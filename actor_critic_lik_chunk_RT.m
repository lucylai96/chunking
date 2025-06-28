function [lik,latents] = actor_critic_lik_chunk_RT(x,data)

%{
    Likelihood function for the actor critic process model that minimizes
    I(S;A|S_t-1)
    
    Called by: fit_models()
%}
agent.m = data.m;

% fixed
agent.t0 = 250;
agent.sigma = 0.5;
agent.cost = 1;

if contains(agent.m, 'fixed_cond')
    agent.lrate_e = 0.1;
    agent.lrate_r = 0.1;
    agent.lrate_beta = 0;
    agent.beta = [x(1) x(6) x(7) x(8)];
    agent.lrate_theta = x(2);
    agent.lrate_V = x(3);
    agent.b1 = x(4);
    agent.b2 = x(5);
    agent.lrate_p = x(9);

elseif contains(agent.m, 'adaptive_cond')
    agent.C = x(1);
    agent.beta0 = x(2);
    agent.lrate_theta = x(3);
    agent.lrate_V = x(4);
    agent.b1 = x(5);
    agent.b2 = x(6);
    agent.lrate_beta = x(7);
    agent.lrate_e = 0.01;
    agent.lrate_p = 0.01;
    agent.lrate_r = 0.01;

end
if ~isfield(agent, 'beta')
    agent.beta = agent.beta0;
end

%cond = {{'Ns4,random_train'}, {'Ns4,structured_train', 'Ns4,structured_test','Ns4,random_test'},...
%    {'Ns6,random_train'}, {'Ns6,structured_train', 'Ns6,structured_test','Ns6,random_test'}};
cond = {{'Ns4,random_train'}, {'Ns4,structured_train'}, {'Ns6,random_train'}, {'Ns6,structured_train'}};

lik = 0; latents.lik = zeros(1,length(cond));

for block = 1:length(cond)
    ix = [];
    expCond = cond{block};
    for c = 1:length(expCond)
        ix = [ix find(strcmp(data.cond, expCond{c}))'];
    end
    reward = data.r(ix);
    action = data.a(ix);
    state = data.s(ix);
    rt = data.rt(ix);
    setsize = length(unique(state));         % number of distinct states
    nA = max(unique(action));                % number of distinct actions
    theta = zeros(setsize,nA);               % policy parameters
    V = zeros(setsize,1)+0.01;               % state values
    p = ones(setsize,nA); p = p./sum(p(1,:)); % conditional default policy
    rho0 = 0;
    if contains(agent.m,'fixed')
        beta = agent.beta(block);
    else
        beta = agent.beta;
    end
    ecost = 0;
    rho = 0;
    g = [];
    lik_choice = 0;
    lik_rt = zeros(1,length(state));

    if nargout > 1                      % if you want to collect the latents
        ii = find(ix);
    end

   if contains(expCond{1},'Ns4,structured')
        p = [1 3 3 3;
            3 1 1 1;
            1 3 3 3;
            1 3 3 3];

    elseif contains(expCond{1},'Ns6,structured')
        p = [3 3 3 1 3 3;
            3 3 3 1 3 3;
            3 3 3 1 3 3;
            3 3 3 1 3 3;
            1 1 1 3 1 1;
            3 3 3 1 3 3];
   end
    p = p./sum(p,2);

    for t = 1:length(state)
        s = state(t); a = action(t); r = reward(t);

        if isnan(a)
            policy(t,:) = NaN;
            continue;
        end

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
        cost = logpolicy(a) - log(prior(a));   % policy complexity cost

        if logpolicy(a)<-10
            logpolicy(a) = -10;
        end

        lik_choice = lik_choice + logpolicy(a);
        lik_rt(t) = (log(rt(t)) - log(agent.t0 + agent.b1*abs(cost) + agent.b2*entropy))^2;

        rpe = beta*r - cost - V(s);
        rho = rho + agent.lrate_r*(r-rho);             % avg reward update
        ecost = ecost + agent.lrate_e*(cost-ecost);    % policy cost update

        chosen = a; idxs = 1:nA; unchosen = idxs(idxs~=chosen);
        g(:,chosen) = beta*(1-policy(chosen));       % policy gradient for chosen actions
        g(:,unchosen) = beta*(-policy(unchosen));    % policy gradient for unchosen actions
        theta(s,:) = theta(s,:) + agent.lrate_theta*rpe*g;  % policy parameter update
        V(s) = V(s) + agent.lrate_V*rpe;

        if agent.lrate_beta > 0
            beta = beta + agent.lrate_beta*(agent.C-ecost);
        end
        ecost = max(ecost,0);
        beta = max(0,min(beta,10));

        % default policy update
        if agent.lrate_p > 0 && t>1
            p(state(t-1),:) = p(state(t-1),:) + agent.lrate_p*(policy - p(state(t-1),:));
            p = p./sum(p,2);  % marginal update
        end
        rho0 = rho;

        if nargout > 1                                          % if you want to collect the latents
            latents.rpe(ii(t)) = rpe;
            latents.ecost(ii(t)) = ecost;
            latents.beta(ii(t)) = beta;
        end
        n(c) = length(state);
    end % each condition

    lik_rt(lik_rt<-10) = -10;
    lik_rt(lik_rt>1e5) = 1e5;
    lik = lik + lik_choice + sum(n)*log(1/(sqrt(2*pi)*agent.sigma))-(1/(2*agent.sigma^2))*sum(lik_rt(:));
    latents.lik(block) = latents.lik(block) + lik_choice + sum(n)*log(1/(sqrt(2*pi)*agent.sigma))-(1/(2*agent.sigma^2))*sum(lik_rt(:));

end % block end

end
