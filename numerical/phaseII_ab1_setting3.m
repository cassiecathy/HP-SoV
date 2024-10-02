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


%% IC model settings
nstage = 3;
ntime = [50,50,50];
nsensor = [4,4,4];
npc = [2,2,2];


B = cell(1,nstage-1);

sele = 1:3:3*(npc(1)-1)+1;
knot = max(sele)+3;
tmp = bspline_basismatrix(3,linspace(0,1,knot),linspace(0,1,ntime(1)+2));
tmp1 = tmp(:,sele);
tmp1(1,:) = [];
tmp1(end,:) = [];
B{1} = tmp1;
B{2} = tmp1;
B{3} = tmp1;

A = cell(1,nstage-1);
C = cell(1,nstage-1);
A{1} = [0.6   0.2;
    0.2  0.6];

A{2} = [0.8   0;
    0  0.8];
    
C{1} = [0.8 0.2 0 0 ;
    0.2 0.8 0 0 ;
    0 0 0.8 0.2 ;
    0 0 0.2 0.8 ];
C{2} = [0.8 0.2 0 0 ;
    0.2 0.8 0 0 ;
    0 0 0.8 0.2 ;
    0 0 0.2 0.8 ];

Ts = cell(1,nstage);
for s = 1:nstage
    Ts{s} = eye(ntime(s));
    for i = 1:(ntime(s)-1)
        for j = (i+1):ntime(s)
            Ts{s}(i,j) = (0.2)^(abs(i-j));
            Ts{s}(j,i) = Ts{s}(i,j);
        end
    end
end

sigma1 = 1e-3*ones(nstage,1);
sigma2 = 1e-3*ones(nstage-1,1);
M1 = [1 0 1 0.5 ;
      0 1 0.5 0.5 ];
V1 = gallery('randcorr',npc(1)*nsensor(1)) * 1e-5;


model.B = B;
model.A = A;
model.C = C;
model.Ts = Ts;
model.sigma1 = sigma1;
model.sigma2 = sigma2;
model.M1 = M1;
model.V1 = V1;


%% OC monitoring settings
ab_senario = 1;
ab_sensor = [1,2];
ab_stage = 1;
ab_sam = 1;
ARL0 = 200;
options.npc = npc;
options.lambda = 50;
options.gamma = 0.1;
options.isbreak = 1;
options.ab_stage = ab_stage;
limit = Inf;

mean_shift = [0:5:40]*10^-3;
test_shift = 1:length(mean_shift);

nIC = 100;
nOC = 1000;
repNum_ic = 2000;
repNum_oc = 1000;

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
[UCL_mfs,ARL0_mfs] = FindCL(T2test_mfs_ic,ARL0,ab_sam);
options.limit = UCL_mfs;
toc;

%%% get ARL ----------
runlen = zeros(1,repNum_oc);
T2test_mfs_oc = zeros(repNum_oc,nOC);

for j = test_shift
    tic;
    shift = mean_shift(j);
    parfor s = 1:repNum_oc
        [ocY,icY] = OCgen(model,nIC,nOC,ab_senario,shift,ab_sensor,ab_stage,ab_sam);
        [runlen(s),T2test_mfs_oc(s,:)] = fssmMon(ocY,icY,options);
    end
    ARL1 = mean(runlen-ab_sam+1);
    SDARL1 = std(runlen-ab_sam+1);
    fname = ['E:\lpy\HFSSM\2 code_clear_update\ARL1_Setting3\ab',num2str(ab_senario),'shift',num2str(j)];
    save(fname,'ARL1','SDARL1');   
    toc;
end

