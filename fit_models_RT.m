function [results,bms_results] = fit_models_RT(models, data, save_mat)
%{
    models = {'fixed'};
    models = {'nocost','fixed','adaptive'};
    [results,bms_results] = fit_models_RT(models, data, 1)

    Fit data to the models you specify.

    no_cost / no_cost_chunk (5 free params)
        agent.lrate_theta:  actor learning rate
        agent.lrate_V:      critic learning rate

    fixed / fixed_chunk (7 free params)
        agent.beta:         fitted value of beta
        agent.lrate_theta:  actor learning rate
        agent.lrate_V:      critic learning rate
        agent.lrate_p:      learning rate for marginal action probability

    example: all 4 models input:
        models = {'no_cost','no_cost_chunk','fixed','fixed_chunk'};
        models = {'fixed','adaptive'};
        [results,bms_results] = fit_models_RT(models, data, save_mat)

%}

if nargin < 3
    save_mat = 0;
end
addpath('/Users/lucy/Library/CloudStorage/GoogleDrive-lucylai.lxl@gmail.com/My Drive/Harvard/Projects/mfit');

if nargin < 2; load('actionChunk_data.mat'); end
idx = 1;

for i = 1:length(models)
    clear param;
    m = models{i};
    minlr = 0.01;
    maxlr = 0.99;
    for i = 1:length(data)
        data(i).m = m;
    end

    if contains(m, 'nocost')
        param(1) = struct('name','beta0','logpdf',@(x) 5,'lb',0.01,'ub',30,'label','\beta');
        param(2) = struct('name','lrate_theta','logpdf',@(x) 0.5,'lb',minlr,'ub',maxlr,'label','lrate_{\theta}');
        param(3) = struct('name','lrate_V','logpdf',@(x) 0.5,'lb',minlr,'ub',maxlr,'label','lrate_V');
        param(4) = struct('name','A','logpdf',@(x) 100,'lb',0,'ub',500,'label','A');
        param(5) = struct('name','b_A','logpdf',@(x) 100,'lb',0,'ub',300,'label','bound-A');

    elseif contains(m, 'fixed') % regular & conditional policy compression
        param(1) = struct('name','beta1','logpdf',@(x) 10,'lb',1,'ub',10,'label','\beta1');
        param(2) = struct('name','lrate_theta','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_{\theta}');
        param(3) = struct('name','lrate_V','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_\rho');
        param(4) = struct('name','b1','lb',1,'ub',500,'logpdf',@(x) 0,'label','b1');
        param(5) = struct('name','b2','lb',1,'ub',500,'logpdf',@(x) 0,'label','b2');
        param(6) = struct('name','beta2','logpdf',@(x) 10,'lb',1,'ub',10,'label','\beta2');
        param(7) = struct('name','beta3','logpdf',@(x) 10,'lb',1,'ub',10,'label','\beta3');
        param(8) = struct('name','beta4','logpdf',@(x) 10,'lb',1,'ub',10,'label','\beta4');
        param(9) = struct('name','lrate_p','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_p');

    elseif contains(m, 'adaptive') % regular & conditional policy compression
        param(1) = struct('name','C','logpdf',@(x) 0.7,'lb',0.01,'ub',3,'label','C');
        param(2) = struct('name','beta0','logpdf',@(x) 1,'lb',1,'ub',10,'label','\beta');
        param(3) = struct('name','lrate_theta','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_{\theta}');
        param(4) = struct('name','lrate_V','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_V');
        param(5) = struct('name','b1','lb',1,'ub',600,'logpdf',@(x) 0,'label','b1');
        param(6) = struct('name','b2','lb',1,'ub',600,'logpdf',@(x) 0,'label','b2');
        %param(7) = struct('name','lrate_beta','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_\beta');
        %param(5) = struct('name','lrate_r','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_\rho');
        param(7) = struct('name','lrate_e','logpdf',@(x) 0.5,'lb',0.01,'ub',0.99,'label','lrate_e');
        %param(7) = struct('name','t0','lb',100,'ub',400,'logpdf',@(x) 0,'label','t0');
    end

    if contains(m, 'cond')
        likfun = @actor_critic_lik_chunk_RT;
    else
        likfun = @actor_critic_lik_RT;
    end
    results(idx) = mfit_optimize(likfun,param,data);
    idx = idx + 1;
end

bms_results = mfit_bms(results);

if save_mat
    save('models_RT_adaptive6.mat','results','bms_results'); % models
end
end