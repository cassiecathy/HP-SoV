function [runlen,T2,stageoc] = fssmMon(ocY,icY,User_options)
% this is the code for Phase I learning and Phase II monitoring based on a
% heterogeneous functional state space model (HFSSM):
%                        Ys = Bs*Xs + Es       s = 1,...,S
%                        Xs+1 = As*Xs*Cs + Ws  s = 1,...,S-1
% It assumes X1~anorm(M1,V1)   Es~anorm(0,Rs)	Ws~anorm(0,Qs)
%            Bs(s=1,...,S) are orthogonal and smooth
% 
% Required input arguments:
%
%     icY:      Input in-control data. 1 x nstage cell;
%               For each cell, it is a ns*ps*nIC data array; ps variables, 
%               ns time points, and nIC samples. Note that ns and ps differ for 
%               different stages s = 1,...,S.
%               Data Types - double
% 
%     ocY:      Input out-of-control data. 1 x nstage cell;
%               For each cell, it is a ns*ps*nOC data array; ps variables, 
%               ns time points, and nOC samples. Note that ns and ps differ for 
%               different stages s = 1,...,S.
%               Data Types - double
% 
% Optional input arguments:
%       options:  Tuning parameters. Struct.
% 
%           options.npc: The dimension of the latent state variable Xs. By default, it is
%           determined according to the proportion of total explained variance. Double. 
% 
%           options.norm: Logical which specifices whether rotate the learned state variables 
%           according to the known state values as options.X_gen. Possible values are 
%           false (default) or true.
%               Example  - options.norm = true
%           Remark: If options.norm = true, the user must input the true latent state variables 
%           as options.X_gen, which is a 1 x nstage cell; For each cell, it
%           is a qs*ps*nIC array; qs states, ps variables, and nIC samples.
%
%           options.lambda: Double which controls how strict the smoothness 
%           penalty is enforced. 
%               Example  - options1.lambda = 0 (default)
%               Example  - options1.lambda = 0.5
% 
%           options.limit: The predetermined upper control limit which controls the
%           Type I monitoring error. Double.
%           Remark: when IC sample size is suffiecnt, we suggest use Monte
%           Carlo to set the UCL such that the average IC profile run
%           length equals to 200.     
% 
%           options.gamma: The EWMA tuning parameter between 0 and 1. A small value means 
%           we have strong ability of remembering the old samples, which helps accumulate
%           small shift and therefore improves the detection ability. On the contrary, 
%           a large value puts more emphases on the recent samples and is able to detect
%           large shift more quickly. To make a trade-off between both sides, we set the 
%           deault value as 0.1 by default. Double.
% 
% 
% Output:
%
%       runlen: double which marks the sample number when the OC alaram
%       is striggered. If no alarm is tirggered, it equals to the total
%       sample number of ocY, which is nOC.
%
%       stageoc: double which marks the stage number where the OC alaram
%       is striggered. If no alarm is tirggered, it equals to the maximum
%       stage of ocY, which is nstage.
% 
%       T2: 1*nOC vector which records the T2 statistics with EWMA. Double. 

%% Beginning of code 
% Input parameters checking
if nargin<2
    error('Error: Input data is missing.');
end

nstage = size(ocY,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
nIC = size(icY{1},3);
for s = 1:nstage
    [ntime(s),nsensor(s),nOC] = size(ocY{s});
end

% The default value 
options = struct('lambda',0,'limit',Inf,'isbreak',1,'gamma',0.1,'ab_stage',1:nstage);
% Input the user setting
if nargin == 3
    if isfield(User_options, 'npc')
        options.npc = User_options.npc;
    end
    if isfield(User_options, 'lambda')
        options.lambda = User_options.lambda;
    end
    if isfield(User_options, 'limit')
        options.limit = User_options.limit;
    end
    if isfield(User_options, 'isbreak')
        options.isbreak = User_options.isbreak;
    end
    if isfield(User_options, 'gamma')
        options.gamma = User_options.gamma;
    end
    if isfield(User_options, 'ab_stage')
        options.ab_stage = User_options.ab_stage;
    end
end




%% Phase I learning via EM algorithm
model = fssmEm(icY,options);
% fssmPre(icY,model,1);% show prediction by HFSSM

B = model.B;
BB = cell(1,nstage);
for s = 1:nstage
    BB{s} = kron(eye(nsensor(s)),B{s});
end


%% forecast via Kalman filter
[xp2,Vp2] = Kalmanfilter(model,icY);
[xp,~] = Kalmanfilter(model,ocY);
yp = cell(1,nstage);
yp2 = cell(1,nstage);
e = cell(1,nstage);
e2 = cell(1,nstage);
Z = cell(1,nstage);
Phi = cell(1,nstage);
for s = 1:nstage
    icY{s} = reshape(icY{s},[ntime(s)*nsensor(s),nIC]);
    ocY{s} = reshape(ocY{s},[ntime(s)*nsensor(s),nOC]);
    yp2{s} = BB{s}*xp2{s};
    yp{s} = BB{s}*xp{s};
    e{s} = yp{s}-ocY{s};
    e2{s} = yp2{s}-icY{s};
    e{s} = e{s} - mean(e2{s},2);
    Z{s} = zeros(ntime(s)*nsensor(s),1);
    Phi{s} = inv(BB{s}*mean(Vp2{s},3)*BB{s}'+ kron(model.sigma1(s)*eye(nsensor(s)),model.Ts{s}));
end


%% Phase II monitoring by group EWMA control chart
gamma = options.gamma;
isbreak = options.isbreak;
limit = options.limit;
ab_stage = options.ab_stage;
runlen = nOC;
stageoc = nstage;
Test = zeros(nstage,nOC);
T2 = zeros(1,nOC);
for i = 1:nOC
    for s = ab_stage
        Z{s} = gamma * e{s}(:,i) +(1-gamma)*Z{s};
        Test(s,i) = (2-gamma)/gamma/(1-(1-gamma)^(2*i))*(Z{s}'*Phi{s}*Z{s})/length(Z{s});
    end
    T2(i) = max(Test(:,i));
    if T2(i)>limit
        runlen = i;stageoc = find(Test(:,i)==max(Test(:,i)));
        if isbreak == 1
            break;
        end
    end

end


end