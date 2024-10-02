% this is the code for Phase I learning of a functional state space model
%                        Yj = muj + B*Xj + Ej
%                        Xj+1 = Aj*Xj*Cj + Wj
% input: set the model: model0
% output: model1 without smooth penalty and orthognal constraint, 
% model2 with smooth penalty and orthognal constraint

% you can adjust the following parameters:
%   lambda           - smoothing parameter
%   nsam             - number of samples

% adds to path
addpath(genpath('E:\0研究\03 Functional SSM\2 code_clear_update\case'));
addpath(genpath('E:\0研究\03 Functional SSM\2 code_clear_update\utilities'));
clear;
close all;
load('diffusioncase.mat')
% load('model_fssm.mat')
% plotY(yic);



nstage = size(yic,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(yic{s});
end

yic_norm = cell(1,nstage);
yoc_norm = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        for j = 1:nsensor(s)
            temp1 = yic{s}(:,j,i);
            yic_norm{s}(:,j,i) = (temp1-mean(temp1))./std(temp1);
            if i <11
            temp2 = yoc{s}(:,j,i);
            yoc_norm{s}(:,j,i) = (temp2-mean(temp1))./std(temp1);
            end
        end
    end
end
plotY(yic_norm);
plotY(yoc_norm);

%% split
yic_norm_train = cell(1,3);
yic_norm_test = cell(1,3);
train_index = randperm(38, 28);
test_index = setdiff(1:38, train_index);
for s = 1:3
    yic_norm_train{s} = yic_norm{s}(:,:,train_index);
    yic_norm_test{s} = cat(3,yic_norm{s}(:,:,test_index),yoc_norm{s});
end



%% EM learning and likelihood
options.lambda = 50;
[model_fssm, llh] = fssmEm(yic_norm_train,options);

figure;
plot(llh,'r-o','LineWidth',1,'MarkerFaceColor','r','MarkerSize',2);
xlabel('Iteration');ylabel('Loglikelihood');
set(gca, 'Fontname', 'Times New Roman','FontSize',10);


%% prediction and see the OSFE
isshow = 1;
yp_fssm = fssmPre(yic_norm_train,model_fssm,isshow);

err = cell(1,nstage);
for s = 1:nstage
    err{s} = yic_norm_train{s} - yp_fssm{s};
%     err{s} = reshape(err{s},[ntime(s)*nsensor(s),nsam])';
end
save('E:/0研究/03 Functional SSM/2 code_clear_update/case/case_result','err','yic_norm_train');



%% monitor
options2.gamma = 0.1;
options2.isbreak = 0;
limit = Inf;

[~,T2test_mfs_ic] = fssmMon(yic_norm_train,yic_norm_test,model_fssm,limit,options2,2);
UCL_mfs = log(max(T2test_mfs_ic(1:10))*1.2)

%% plot chart
Test = log(T2test_mfs_ic);
idxx = 1:length(Test);
outpoint = 12;

figure('position',[200,200,200,180]);
plot(idxx,Test,'color',[0.8 0.8 0.8],'Marker','o','MarkerFaceColor',[0.8 0.8 0.8],'MarkerSize',3);hold on;
plot([10,10],[0,4],'k--','linewidth',1);hold on;
h1 = plot(1:outpoint,Test(1:outpoint),'ko-','MarkerFaceColor','k','MarkerSize',3);hold on;
h2 = plot([0,length(Test)],[UCL_mfs,UCL_mfs],'r--','linewidth',1);hold on;
idddd = 13:20;
iddddy = Test(13:20);
for i=1:8
tx=num2str(2);
text(idddd(i)-0.5,iddddy(i),tx);
end
ylim([0,4]);
leg = legend([h1,h2],{'log(MT_i)','UCL'},'Fontname', 'Times New Roman','FontSize',8,'Location','northwest');
leg.ItemTokenSize = [15,3];
xlabel('Observations');
title('HFSSM Control Chart')
set(gca, 'Fontname', 'Times New Roman','FontSize',8);













