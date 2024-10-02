% this is the code for Phase I learning and Phase II monitoring based on a
% heterogeneous functional state space model (HFSSM):
%                        Ys = Bs*Xs + Es       s = 1,...,S
%                        Xs+1 = As*Xs*Cs + Ws  s = 1,...,S-1
% It assumes X1~anorm(M1,V1)   Es~anorm(0,Rs)	Ws~anorm(0,Qs)
%            Bs(s=1,...,S) are orthogonal and smooth

% % adds to path
addpath(genpath('E:\0研究\03 Functional SSM\2 code_clear_update\HFSSM'));
addpath(genpath('E:\0研究\03 Functional SSM\2 code_clear_update\utilities'));
clear;
close all;

%% Artificial data
% rng('default')
rng(11);

nstage = 3;
ntime = [50,64,80];
nsensor = [4,4,6];
npc = [2,2,3];

B = cell(1,nstage-1);
tmp2 = fourierbasis(npc(1)+1,ntime(1));
B{1} = tmp2(:,1:npc(1));
tmp3 = WTortho(ntime(2),'Vaidyanathan',4,2);
B{2}= tmp3(1:npc(2),:)';
sele = 1:3:3*(npc(3)-1)+1;
knot = max(sele)+3;
tmp = bspline_basismatrix(3,linspace(0,1,knot),linspace(0,1,ntime(3)+2));
tmp1 = tmp(:,sele);
tmp1(1,:) = [];
tmp1(end,:) = [];
B{3} = tmp1;
% plotB(B);% show bases of the three-stage processes

A = cell(1,nstage-1);
C = cell(1,nstage-1);
A{1} = [0.6   0.2;
    0.2  0.6];
A{2} = [0.8   0 ;
        0  0.8 ;
        0.4  0.4];
C{1} = [0.8 0.2 0 0 ;
    0.2 0.8 0 0 ;
    0 0 0.8 0.2 ;
    0 0 0.2 0.8 ];
C{2} = [0.8 0.2 0 0 0.5 0 ;
    0.2 0.8 0 0 0 0.5;
    0 0 0.8 0.2 0.5 0;
    0 0 0.2 0.8 0 0.5];

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

% sigma1 = 1e-3*ones(nstage,1);
% sigma2 = 1e-3*ones(nstage-1,1);
% M1 = [1 0 1 0.5 ;
%       0 1 0.5 0.5 ];
% 
% temp = rand(npc(1)*nsensor(1));
% V1 = temp*temp'* 1e-5 + eye(npc(1)*nsensor(1))* 1e-3;

sigma1 = 1e-3*ones(nstage,1);
sigma2 = 1e-3*ones(nstage-1,1);
M1 = [1 0 1 0.5 ;
      0 1 0.5 0.5 ];

V1 = gallery('randcorr',npc(1)*nsensor(1)) * 1e-3;

model0.B = B;
model0.A = A;
model0.C = C;
model0.Ts = Ts;
model0.sigma1 = sigma1;
model0.sigma2 = sigma2;
model0.M1 = M1;
model0.V1 = V1;


%% estimate
nsam = 100;
[y_train,X_gen] = genfssm(model0,nsam);

options.norm = 1;
options.X_gen = X_gen;
options.npc = npc;

lambda = 100;%[50,100,150,200]
options.lambda = lambda;

[model_fssm, llh] = fssmEm(y_train,options);

%% fit
b_true = model0.B;
b_fssm = model_fssm.B;
b_sum = [b_true;b_fssm];

nsam_test = 100;
y_test = genfssm(model0,nsam_test);
isshow = 0;
yp_fssm = fssmPre(y_test,model_fssm,isshow);
y_sum = [y_test;yp_fssm];

[in_angle,in_mse,in_stse,in_mpcc,in_stpcc,in_lambda] = index(b_sum,y_sum);
