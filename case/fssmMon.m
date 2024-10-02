function [runlen,T2,stageoc] = fssmMon(icY,ocY,model,limit,options,ab_stage)
% input:ocY ntime*nsensor*nstage*nsam
% output: T-square test statistics 1*nsam

gamma = options.gamma;
isbreak = options.isbreak;

nstage = size(ocY,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
m0 = size(icY{1},3);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(ocY{s});
end
B = model.B;
BB = cell(1,nstage);
for s = 1:nstage
    BB{s} = kron(eye(nsensor(s)),B{s});
end


%% forecast
[~,~,~,~,~,xp2,Vp2] = Estep(model,icY);
[~,~,~,~,~,xp,Vp] = Estep(model,ocY);
yp = cell(1,nstage);
yp2 = cell(1,nstage);
e = cell(1,nstage);
e2 = cell(1,nstage);
Z = cell(1,nstage);
Phi = cell(1,nstage);
for s = 1:nstage
    icY{s} = reshape(icY{s},[ntime(s)*nsensor(s),m0]);
    ocY{s} = reshape(ocY{s},[ntime(s)*nsensor(s),nsam]);
    yp2{s} = BB{s}*xp2{s};
    yp{s} = BB{s}*xp{s};
    e{s} = yp{s}-ocY{s};
    e2{s} = yp2{s}-icY{s};
    e{s} = e{s} - mean(e2{s},2);
    Z{s} = zeros(ntime(s)*nsensor(s),1);
%     Phi{s} = inv(diag(diag(cov(e2{s}'))));
    Phi{s} = inv(BB{s}*mean(Vp2{s},3)*BB{s}'+ model.sigma1*eye(ntime(s)*nsensor(s)));
end


%% EWMA on sample
runlen = nsam;
stageoc = nstage;
Test = zeros(nstage,nsam);
T2 = zeros(1,nsam);
for i = 1:nsam
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