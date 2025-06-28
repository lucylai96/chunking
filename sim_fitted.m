function simdata = sim_fitted(model, data, results)

%{
    Simulate the agent with the fitted model parameter for the set size
    manipulation experiment.

    USAGE: simdata = sim_fitted()
%}

prettyplot

for s = 1:length(data)
    agent.m = model;

    for k = 1:length(results.param)
        agent.(results.param(k).name) = results.x(s,k);
    end

    if contains(model,'nocost')
        % fixed parameters
        agent.b = agent.b_A+agent.A;
        agent.b2 = agent.b2_A+agent.A;
        agent.sv = 0.1;
        agent.eta = 1;

        simdata(s) = actor_critic_nc(agent, data(s));

    elseif contains(model,'rlwm')
        % fixed parameters
        agent.b = agent.b_A+agent.A;
        agent.b2 = agent.b2_A+agent.A;
        agent.sv = 0.1;
        agent.eta = 1;
        agent.beta_rl = 50;
        agent.beta_wm = 50;

        simdata(s) = actor_critic_rlwm(agent, data(s));

    elseif contains(model,'fixed')
        agent.lrate_e = 0.1;
        agent.lrate_r = 0.1;
        agent.t0 = 250;
        agent.sigma = 0.5;
        agent.cost = 1;
        agent.lrate_beta = 0;
        agent.lrate_p = 0.1;
        agent.beta = [agent.beta1 agent.beta2 agent.beta3 agent.beta4];

        if contains(model,'cond')  % fixed conditional policy compression
            simdata(s) = actor_critic_sim_chunk(agent, data(s));
        else  % fixed conditional policy compression
            simdata(s) = actor_critic_sim(agent, data(s));
        end

    elseif contains(model,'adaptive')
        agent.t0 = 250;
        agent.sigma = 0.5;
        agent.cost = 1;
        %agent.lrate_theta = 0.01;
        %agent.lrate_V = 0.01;
        %agent.b1 = 200;
        %agent.b2 = 170;
        %agent.C = 0.5;
        %agent.beta0 = 1;

        %agent.lrate_theta = 0.1;
        %agent.lrate_V = 0.1;
        %agent.lrate_beta = 0.1;
        %agent.C = agent.C-0.6;
        %agent.C = max(0.1,agent.C);
        agent.lrate_e = 0.01;   % cost learning rate
        agent.lrate_p = 0.01;    % default learning rate
        agent.lrate_r = 0.01;    % reward learning rate
        agent.b1 = agent.b1+50;
        if contains(model,'cond') % adaptive conditional policy compression
            simdata(s) = actor_critic_sim_chunk(agent, data(s));
        else % adaptive policy compression
            simdata(s) = actor_critic_sim(agent, data(s));
        end
    end % switch on model
end % for each subject
end

