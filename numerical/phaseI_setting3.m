clear;
close all;

%% Artificial data
rng('default')
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
sigma2 = 1e-5*ones(nstage-1,1);
M1 = [1 0 1 0.5 ;
      0 1 0.5 0.5 ];

temp = rand(npc(1)*nsensor(1));
V1 = temp*temp'* 1e-5 + eye(npc(1)*nsensor(1))* 1e-4;


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
% plotY(y_train);

options.norm = 1;
options.X_gen = X_gen;
options.npc = npc;
options.lambda = 100;

model_fssm = fssmEm(y_train,options);



%% fit
b_true = model0.B;
b_fssm = model_fssm.B;
b_sum = [b_true;b_fssm];

nsam_test = 100;
y_test = genfssm(model0,nsam_test);
isshow = 0;
yp_fssm = fssmPre(y_test,model_fssm,isshow);
y_sum = [y_test;yp_fssm];

[in_angle,in_mse,in_stse,in_mpcc,in_stpcc] = index(b_sum,y_sum);



