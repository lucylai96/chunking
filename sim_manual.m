function simdata = sim_manual(model,data)

%{
    Simulate the agent with the fitted model parameter for the set size
    manipulation experiment.

    USAGE: simdata = sim_fitted()

    Called by: analyze_simdata_exp1()
%}

if nargin < 1; load data.mat; end

for s = 1:length(data)
    %% NOTES %%
    % need agent.lrate_p > 0 to have any differences in RT between
    % structured and random train (important for joint choice + RT model) 
    % also need lrate_p > 0 to have any increase in RT for Ns=6
    % seems like lrate_p will be deciding factor in joint model,
    % choice-only model would favor no chunk

    % betas are important for how much reliance on marginal, but betas
    % themselves can distinguish accuracy between conditions (when lrate_p = 0)
    % diff betas for structured and random can contribute to cost in RT
    % simulation and produce differences in RT 

    % for the same beta, complexity should be less in structured

    agent.m = model;

    % set parameters
    agent.t0 = 300;
    agent.sigma = 0.5;
    agent.cost = 1;

    agent.V = 0.8;
    agent.C = 0.7; 
    agent.beta0 = 1;
    agent.lrate_theta = 0.1;
    agent.lrate_V = 0.1;
    agent.lrate_beta = 0;

    agent.b1 = 200;
    agent.b2 = 288;
    agent.lrate_e = 0.05;   % cost learning rate
    agent.lrate_r = 0.05;    % reward learning rate
    agent.lrate_p = 0;    % default learning rate

if contains(model,'fixed')
        %agent.lrate_e = 0.1;
        %agent.lrate_r = 0.1;
        agent.lrate_beta = 0;
        agent.beta = [4 5 6 7];

        if contains(model,'cond')  % fixed conditional policy compression
            simdata(s) = actor_critic_sim_chunk(agent, data(s));
        else  % fixed conditional policy compression
            simdata(s) = actor_critic_sim(agent, data(s));
        end
    elseif contains(model,'adaptive')
        %agent.lrate_e = 0.1;
        %agent.lrate_beta = 0.3;    % lower allows it to gain accuracy faster
        
        if contains(model,'cond') % adaptive conditional policy compression
            simdata(s) = actor_critic_sim_chunk(agent, data(s));
        else % adaptive policy compression
            simdata(s) = actor_critic_sim(agent, data(s));
        end
end

end
