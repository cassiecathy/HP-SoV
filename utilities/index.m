function [angle,mse,stse,mpcc,stpcc,chose_lambda] = index(b_sum,y_sum)
nmethod1 = size(b_sum,1)-1;
nmethod2 = size(y_sum,1)-1;
angle = zeros(1,nmethod1);
mse = zeros(1,nmethod2);
mpcc = zeros(1,nmethod2);
stse = zeros(1,nmethod2);
stpcc = zeros(1,nmethod2);

b_true = b_sum(1,:);
b_fssm = b_sum(2,:);
y_true = y_sum(1,:);

nstage = size(b_true,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
df = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(y_true{s});
    [~,npc(s)] = size(b_true{s});
    [~,df(s)] = size(b_fssm{s});
end
y_sumvec = cell(nmethod2+1,nstage);
y_truevec = cell(1,nstage);
for s = 1:nstage
    for i = 1:nmethod2+1 
        y_sumvec{i,s} = reshape(y_sum{i,s},[ntime(s)*nsensor(s),nsam]);
    end
    y_truevec{s} = reshape(y_true{s},[ntime(s)*nsensor(s),nsam]);
end


for i = 1:nmethod1
    b_temp = b_sum(i+1,:);
    temp = [];
    for s = 1:nstage
        for j = 1:npc(s)
            temp0 = abs(acos((b_true{s}(:,j)'./norm(b_true{s}(:,j)))*(b_temp{s}(:,j)./norm(b_temp{s}(:,j)))));
            temp0 = min(temp0,pi-temp0);
            temp = [temp,temp0];
        end
    end
    angle(i) = mean(temp);
end



for i = 1:nmethod2
    yp_temp = y_sumvec(i+1,:);
    se = zeros(1,nsam);
    for k = 1:nsam
        temp = 0;
        for s = 1:nstage
            temp = temp+(norm(y_truevec{s}(:,k)-yp_temp{s}(:,k)))^2;
        end
        se(k) = temp;
    end
    if i ==1
        chose_lambda = (100/(100-sum(df)))*(sum(se)/(100-sum(df)));
    end
    mse(i) = mean(se);
    stse(i) = std(se);
    
    yp_temp = y_sum(i+1,:);
    pcc = zeros(1,nsam);
    for k = 1:nsam
        temp = [];
        for s = 1:nstage-1
            for j1 = 1:nsensor(s)
                for j2 = 1:nsensor(s+1)
                    temp1 = y_true{s}(:,j1,k)-yp_temp{s}(:,j1,k);
                    temp2 = y_true{s+1}(:,j2,k)-yp_temp{s+1}(:,j2,k);
                    ylen2=length(temp2);%原向量长�?
                    ylen1=length(temp1);%�?终向量的�?求长�?
                    if ~(ylen2 == ylen1)
                        number=ylen2-ylen1;%�?要去除的点的个数
                        number_len=ylen2/number-0.001;%计算等间距去除的步长
                        index=ceil(1:number_len:ylen2);%获取角标
                        temp2(index)=[];%去除对应角标处的�?
                    end
                    temp = [temp,abs(corr(temp1,temp2,'type','pearson'))]; 
                end
            end
        end
        pcc(k) = mean(temp);
    end
    mpcc(i) = mean(pcc);
    stpcc(i) = std(pcc);
    
end



end