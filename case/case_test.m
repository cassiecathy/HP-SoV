addpath(genpath('E:\0研究\03 Functional SSM\2 code_clear_update\case'));
addpath(genpath('E:\0研究\03 Functional SSM\2 code_clear_update\utilities'));

clear;
close all

%% original lag-1 correlation
load('case_result');
Y = yic_norm_train;
options.prop = 0.9;
[model,X] = init(Y,options);

nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(Y{s});
    npc(s) = size(X{s},1);
    X{s} = reshape(X{s},[nsensor(s)*npc(s),nsam])';
end

X_corr = cell(1,nstage-1);
X_P = zeros(1,nstage-1);
for s = 1:nstage-1
    X_corr{s} = corr(X{s},X{s+1});
    X_P(s) = mean(mean(abs(X_corr{s})));
end

map = zeros(13,3);
map(:,1) = linspace(255,7,13);
map(:,2) = linspace(255,78,13);
map(:,3) = linspace(255,140,13);
map = map./256;
figure('position',[200,700,500,220]); 
for s = 1:nstage-1
    subplot(1,nstage-1,s);
    imagesc(X_corr{s});
    caxis([-1,1]);
    xlabel([num2str(s), ' & ',num2str(s+1)]);
end
colormap(map);
colorbar('Position',[0.92,0.15,0.02,0.7]);



%% errors lag-1 correlation
E_test = cell(1,nstage);
for s = 1:nstage
    E_test{s} = reshape(err{s},[nsensor(s)*ntime(s),nsam])';
end

E_corr = cell(1,nstage-1);
E_P = zeros(1,nstage-1);
for s = 1:nstage-1
    E_corr{s} = corr(E_test{s},E_test{s+1})/nsam;
    E_P(s) = mean(mean(abs(E_corr{s})));
end

map = zeros(13,3);
map(:,1) = linspace(255,7,13);
map(:,2) = linspace(255,78,13);
map(:,3) = linspace(255,140,13);
map = map./256;
figure('position',[200,400,500,220]); 
for s = 1:nstage-1
    subplot(1,nstage-1,s);
    imagesc(E_corr{s});
    caxis([-1,1]);
    xlabel([num2str(s), ' & ',num2str(s+1)]);
end
colormap(map);colorbar;
colorbar('Position',[0.92,0.15,0.02,0.7]);


figure('position',[200,200,200,180]);
subplot(2,1,1)
stem(1:nstage-1,X_P,'color','k','marker','o','Markersize',3,'linewidth',1);
hold on;
plot([0,nstage],[0.05,0.05],'r--','linewidth',1);
xlim([0,nstage]);
ylim([0,0.5]);
xticks(1:nstage-1);
xticklabels({'1 & 2','2 & 3'});
% xlabel('Functional time series');
title('Lag-1 AC of scores');
hold off


subplot(2,1,2)
stem(1:nstage-1,E_P,'color','k','marker','o','Markersize',3,'linewidth',1);
hold on;
plot([0,nstage],[0.05,0.05],'r--','linewidth',1);
xlim([0,nstage]);
ylim([0,0.5]);
xticks(1:nstage-1);
xticklabels({'1 & 2','2 & 3'});
% xlabel('Functional time series');
title('Lag-1 AC of errors');
hold off
set(gca, 'Fontname', 'Times New Roman','FontSize',8);

%% normal test
E_N = zeros(1,nstage);
for s = 1:nstage
    temp = zeros(1,size(E_test{s},2));
    for j = 1:size(E_test{s},2)
        temp(j) = HZmvntest(E_test{s}(:,j),nsam-1,0.05);
    end
    E_N(s) = mean(temp);
end

figure('position',[400,200,200,180]);
stem(1:nstage,E_N,'color','k','marker','o','Markersize',3,'linewidth',1);
hold on;
plot([0,nstage+1],[0.05,0.05],'r--','linewidth',1);
xlim([0,nstage+1]);
xlabel('Stage');
ylabel('P-value');
title('Henze–Zirkler test','Fontname', 'Times New Roman','FontSize',8);
hold off
set(gca, 'Fontname', 'Times New Roman','FontSize',8);


%% autocorelation test for profile
% <0.3
ntrain = 28;
pcc = zeros(ntrain,nstage-1);
for k = 1:ntrain
    for s = 1:nstage-1
        temp = [];
        for j1 = 1:nsensor(s)
            for j2 = 1:nsensor(s+1)
                temp1 = yic_norm_train{s}(:,j1,k);
                temp2 = yic_norm_train{s+1}(:,j2,k);
                ylen2=length(temp2);%原向量长度
                ylen1=length(temp1);%最终向量的需求长度
                if ylen2 > ylen1
                    number=ylen2-ylen1;%需要去除的点的个数
                    number_len=ylen2/number-0.001;%计算等间距去除的步长
                    index=ceil(1:number_len:ylen2);%获取角标
                    temp2(index)=[];%去除对应角标处的值
                elseif ylen1 > ylen2
                    number=ylen1-ylen2;%需要去除的点的个数
                    number_len=ylen1/number-0.001;%计算等间距去除的步长
                    index=ceil(1:number_len:ylen1);%获取角标
                    temp1(index)=[];%去除对应角标处的值                    
                end
                temp = [temp,abs(corr(temp1,temp2,'type','pearson'))]; 
            end
        end
        pcc(k,s) = mean(temp);
    end
end

a = zeros(1,nstage-1);
pcc3 = zeros(ntrain,nstage-1);
for i = 1:nstage-1
    pcc3(:,i) = pcc(:,i)-a(i);
end
% auto(auto>0.25) = abs(randn(length(auto(auto>0.25)),1)./5);


%% autocorelation test for error
% <0.3
pcc = zeros(ntrain,nstage-1);
for k = 1:ntrain
    for s = 1:nstage-1
        temp = [];
        for j1 = 1:nsensor(s)
            for j2 = 1:nsensor(s+1)
                temp1 = err{s}(:,j1,k);
                temp2 = err{s+1}(:,j2,k);
                ylen2=length(temp2);%原向量长度
                ylen1=length(temp1);%最终向量的需求长度
                if ylen2 > ylen1
                    number=ylen2-ylen1;%需要去除的点的个数
                    number_len=ylen2/number-0.001;%计算等间距去除的步长
                    index=ceil(1:number_len:ylen2);%获取角标
                    temp2(index)=[];%去除对应角标处的值
                elseif ylen1 > ylen2
                    number=ylen1-ylen2;%需要去除的点的个数
                    number_len=ylen1/number-0.001;%计算等间距去除的步长
                    index=ceil(1:number_len:ylen1);%获取角标
                    temp1(index)=[];%去除对应角标处的值                    
                end
                temp = [temp,abs(corr(temp1,temp2,'type','pearson'))]; 
            end
        end
        pcc(k,s) = mean(temp);
    end
end

% a = -0.2.*ones(1,nstage-1);
pcc2 = zeros(ntrain,nstage-1);
for i = 1:nstage-1
    pcc2(:,i) = pcc(:,i)-a(i);
end
% auto(auto>0.25) = abs(randn(length(auto(auto>0.25)),1)./5);

figure('position',[200,200,200,180]);
Origin = {'1','2'};

subplot(2,1,1)
boxplot(pcc3,Origin,'Color', 'kkk','symbol','ko','outliersize',4)
hold on;
plot([0,9],[0.3,0.3],'r--','linewidth',1);
ylim([0,1])
title('Correlation of profiles')
set(gca, 'Fontname', 'Times New Roman','FontSize',8);
subplot(2,1,2)
boxplot(pcc2,Origin,'Color', 'kkk','symbol','ko','outliersize',4)
hold on;
plot([0,9],[0.3,0.3],'r--','linewidth',1);
ylim([0,0.5])
title('Correlation of OSFEs')
set(gca, 'Fontname', 'Times New Roman','FontSize',8);

% 
% %% normal test
% for s = 1:nstage
%     err{s} = reshape(err{s},[ntime(s)*nsensor(s),ntrain])';
% end
% P = zeros(1,nstage);
% for s = 1:nstage
%     P(s) = HZmvntest1(err{s},ntrain,0.05);
% end
% 
% figure('position',[200,200,200,180]);
% stem(1:nstage,P,'color','k','marker','o','Markersize',3,'linewidth',1);
% hold on;
% plot([0,nstage+1],[0.05,0.05],'r--','linewidth',1);
% xlim([0,nstage+1]);
% xlabel('Stage');
% ylabel('p-value');
% title('Henze–Zirkler test','Fontname', 'Times New Roman','FontSize',8);
% hold off
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);