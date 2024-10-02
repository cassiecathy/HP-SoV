% this is the code for Phase I learning and Phase II monitoring based on a
% heterogeneous functional state space model (HFSSM):
%                        Ys = Bs*Xs + Es       s = 1,...,S
%                        Xs+1 = As*Xs*Cs + Ws  s = 1,...,S-1
% It assumes X1~anorm(M1,V1)   Es~anorm(0,Rs)	Ws~anorm(0,Qs)
%            Bs(s=1,...,S) are orthogonal and smooth

% you can adjust the following parameters:
%   options                       - several setting parameters: npc, norm,
%   lambda, limit, isbreak, gamma... See function 'fssmMon' for their
%   detailed definitions.
%   repNum_ic                     - the number of IC replication(2000,5000)
%   repNum_oc                     - the number of OC replication(1000,2000)
%   nIC                           - traning sample set(50)
%   nOC                           - total run length(1000)

% This is the code under Setting I. For other settings, just change the
% "model" structure. Please refer to the codes of phase I.
% % adds to path
addpath(genpath('E:\lpy\HFSSM\2 code_clear_update\HFSSM'));
addpath(genpath('E:\lpy\HFSSM\2 code_clear_update\utilities'));
clear;
close all;

load('num2_monitoring')
npc = zeros(1,3);
for s = 1:3
    npc(s) = size(model.B{s},2);
end

%% OC monitoring settings
ab_senario = 2;
ab_sensor = [3,4];
ab_stage = 3;
ab_sam = 1;
ARL0 = 200;
options.npc = npc;
options.lambda = 50;
options.gamma = 0.1;
options.isbreak = 1;
options.ab_stage = ab_stage;
limit = Inf;

slope_shift = [0:5:40]*10^-5;
test_shift = 1:length(slope_shift);

nIC = 100;
nOC = 1000;
repNum_ic = 2000;
repNum_oc = 1000;
% [ocY,icY] = OCgen(model,nIC,nOC,ab_senario,0.01,ab_sensor,ab_stage,ab_sam,1);

parpool(40)
% ============================start simulation ============================
% =========================================================================
%%% get UCL --------
shift = 0;
T2test_mfs_ic = zeros(repNum_ic,nOC);
tic;
parfor s = 1:repNum_ic
    [ocY,icY] = OCgen(model,nIC,nOC,ab_senario,shift,ab_sensor,ab_stage,ab_sam);
    [~,T2test_mfs_ic(s,:)] = fssmMon(ocY,icY,options);
end
[UCL_mfs2,ARL0_mfs] = FindCL(T2test_mfs_ic,ARL0,ab_sam);
options.limit = UCL_mfs2;
toc;

%%% get ARL ----------
runlen = zeros(1,repNum_oc);
T2test_mfs_oc = zeros(repNum_oc,nOC);

for j = test_shift
    tic;
    shift = slope_shift(j);
    parfor s = 1:repNum_oc
        [ocY,icY] = OCgen(model,nIC,nOC,ab_senario,shift,ab_sensor,ab_stage,ab_sam);
        [runlen(s),T2test_mfs_oc(s,:)] = fssmMon(ocY,icY,options);
    end
    ARL1 = mean(runlen-ab_sam+1);
    SDARL1 = std(runlen-ab_sam+1);
    fname = ['E:\lpy\HFSSM\2 code_clear_update\ARL1_Setting2\ab',num2str(ab_senario),'shift',num2str(j)];
    save(fname,'ARL1','SDARL1');   
    toc;
end


