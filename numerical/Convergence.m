% this is the code for Phase I learning and Phase II monitoring based on a
% heterogeneous functional state space model (HFSSM):
%                        Ys = Bs*Xs + Es       s = 1,...,S
%                        Xs+1 = As*Xs*Cs + Ws  s = 1,...,S-1
% It assumes X1~anorm(M1,V1)   Es~anorm(0,Rs)	Ws~anorm(0,Qs)
%            Bs(s=1,...,S) are orthogonal and smooth

% % adds to path
addpath(genpath('E:\lpy\HFSSM\2 code_clear\HFSSM'));
addpath(genpath('E:\lpy\HFSSM\2 code_clear\utilities'));
clear;
close all;

rep = 100;
totalsam = [25,50,100,200,500,800,1000];
%[50,80,100,150,200,300,500,800];
% itn = zeros(1,rep);
% itt = zeros(1,rep);
ppdiff = zeros(length(totalsam),8);% 8 parameters
itt = zeros(length(totalsam),rep); % time per iteration
itn = zeros(length(totalsam),rep); % iteration number
tic;
for k = 1:length(totalsam)
    pdiff = zeros(rep,8);
for r = 1:rep

%% IC model settings
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
for s = 1:nstage-1
    [U,S,V] = svd(rand(npc(s+1),npc(s)));
    S(abs(S)>1) = 0.99;
    A{s} = U*S*V';
    [U,S,V] = svd(rand(nsensor(s),nsensor(s+1)));
    S(abs(S)>1) = 0.99;
    C{s} = U*S*V'; 

%     A{s} = rand(npc(s+1),npc(s))*0.5;
%     C{s} = rand(nsensor(s),nsensor(s+1))*0.5;
end

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
M1 = rand(npc(1),nsensor(1));

V1 = gallery('randcorr',npc(1)*nsensor(1)) * 1e-3;

scale = zeros(2,nstage-1);
for s = 1:nstage-1
    D = sort(abs(eig(A{s}'*A{s})),'descend');
    scale(1,s) = sqrt(D(1));
    D = sort(abs(eig(C{s}'*C{s})),'descend');
    scale(2,s) = sqrt(D(1));
end

model0.B = B;
model0.A = A;
model0.C = C;
model0.Ts = Ts;
model0.sigma1 = sigma1;
model0.sigma2 = sigma2;
model0.M1 = M1;
model0.V1 = V1;


%% estimate
nsam = totalsam(k);
[y_train,X_gen] = genfssm(model0,nsam);
% plotY(y_train);

options.norm = 1;
options.X_gen = X_gen;
options.npc = npc;
options.lambda = 50;
options.scale = scale;

    t1 = clock;
    [model_fssm, llh] = fssmEm(y_train,options);
    t2 = clock;
    itt(k,r) = etime(t2,t1)/length(llh);
    itn(k,r) = length(llh);
    % plotB(model_fssm.B);
    pdiff(r,:) = comp(model0,model_fssm);
end

ittt = mean(itt,2);
itnn = mean(itn,2);
ppdiff(k,:) = mean(pdiff,1);
end
toc;
% figure;
% plot(llh,'r-o','LineWidth',1,'MarkerFaceColor','r','MarkerSize',2);
% xlabel('Iteration');ylabel('Loglikelihood');
% set(gca, 'Fontname', 'Times New Roman','FontSize',10);



out_convergence = [totalsam',ppdiff,itnn,ittt];
save('E:/lpy/HFSSM/2 code_clear/HFSSM/convergence_analysis','out_convergence');
