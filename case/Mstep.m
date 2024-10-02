function model1 = Mstep(Y, Ex, Exx, Exy,model0, options)
% Input:
% Y: cell(1,nstage):ntime*nsensor*nsam
%   Ex: E(x_t): cell(1,nstage):(npc*nsensor)*nsam
%   Exx: E[z_tz_t^T]: cell(1,nstage):(npc*nsensor)*(npc*nsensor)*nsam
%   Exy: E[z_tz_{t-1}^T]: cell(1,nstage-1):(npc*nsensor)*(npc*nsensor)*nsam
%   llh: loglikelihood

lambda = options.lambda;
nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(Y{s});
end
for s = 1:nstage
    [~,npc(s)] = size(model0.B{s});
end
model1 = model0;


YX_block = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        YX_block{s}(:,:,i) = Y{s}(:,:,i)*Ex{s}(:,:,i)';
    end
    YX_block{s} = sum(YX_block{s},3);
end
Ex_vec = cell(1,nstage);
Y_vec = cell(1,nstage);
for s = 1:nstage
    Ex_vec{s} = reshape(Ex{s},[npc(s)*nsensor(s),nsam]); 
    Y_vec{s} = reshape(Y{s},[ntime(s)*nsensor(s),nsam]); 
end
YX = cell(1,nstage);
YY = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        YY{s}(:,:,i) = Y_vec{s}(:,i)*Y_vec{s}(:,i)';
        YX{s}(:,:,i) = Y_vec{s}(:,i)*Ex_vec{s}(:,i)';
    end
    YY{s} = sum(YY{s},3);
    YX{s} = sum(YX{s},3);
end


Exx_sum_ten = cell(1,nstage);
Exx_sum = cell(1,nstage);
Exy_sum = cell(1,nstage-1);
Exy_sum_ten = cell(1,nstage-1);
for s = 1:nstage
    Exx_sum{s} = sum(Exx{s},3);
    Exx_sum_ten{s} = reshape(tensor(Exx_sum{s}),[npc(s),nsensor(s),npc(s),nsensor(s)]);
end

for s = 1:nstage-1
    Exy_sum{s} = sum(Exy{s},3);
    Exy_sum_ten{s} = permute(reshape(tensor(Exy_sum{s}),[npc(s+1),nsensor(s+1),npc(s),nsensor(s)]),[3,4,1,2]);
end

%% B sigma1
sigma1 = model0.sigma1;

B = cell(1,nstage);
for s = 1:nstage
    B{s} = zeros(ntime(s),npc(s));
    diffm1 = diffm(ntime(s));
    diffm2 = diffm(ntime(s)-1);
    omega =diffm1'*diffm2'*diffm2*diffm1;

    F = kron(eye(npc(s)),lambda*sigma1*omega) + kron(double(ttt(Exx_sum_ten{s},tensor(eye(nsensor(s))),[2,4],[1,2])),eye(ntime(s)));
    h = reshape(YX_block{s},[],1);
    B{s} = reshape(F\h,ntime(s),npc(s)); 
    
%     F = inv(F);
%     L = kron(eye(npc(s)),model0.B{s});
%     Lam = (L'*F*L)\(L'*F*h-reshape(eye(npc(s)),[],1));
%     B{s} = reshape(F*(h-L*Lam),ntime(s),npc(s));       
end


sigma1 = zeros(1,nstage);
for s = 1:nstage
    BB = kron(eye(nsensor(s)),B{s});
    % R = (YY-B0*YX')/(nstage*nsam);      
    sigma1(s) = trace(YY{s}-BB*YX{s}')/(nsam*ntime(s)*nsensor(s));
end
sigma1 = mean(sigma1);


%% A C sigma2
A = model0.A;
C = model0.C;

for s = 1:nstage-1
    A{s} = double(ttt(Exy_sum_ten{s},tensor(C{s}),[2,4],[1,2]))'/double(ttt(Exx_sum_ten{s},tensor(C{s}*C{s}'),[2,4],[1,2]));
    C{s} = double(ttt(tensor(A{s}'*A{s}),Exx_sum_ten{s},[1,2],[1,3]))\double(ttt(tensor(A{s}'),Exy_sum_ten{s},[1,2],[1,3]));
end

sigma2 = zeros(1,nstage-1);
for s = 1:nstage-1
    F = kron(C{s}',A{s});
    sigma2(s) = trace(Exx_sum{s+1}- 2* F*Exy_sum{s}'+F*Exx_sum{s}*F') / (nsam*nsensor(s+1)*npc(s+1));
end
sigma2 = mean(sigma2);
          


%% M1 , V1 
M1 = mean(Ex{1},3);    
M1_vec = reshape(M1,[],1);
V1 = Exx_sum{1}./nsam - M1_vec*M1_vec';

% V1 = zeros(npc(1)*nsensor(1));
% for i = 1:nsam
% V1 = V1+ Exx{1}(:,:,i)-Ex_vec{1}(:,i)*Ex_vec{1}(:,i)';
% end
% V1 = V1./nsam;
% V1 = recons((V1+V1')/2);

% R = (R+R')/2;
% Q = (Q+Q')/2;
% V1 = (V1+V1')/2;

model1.B = B;
model1.A = A;
model1.C = C;
model1.sigma1 = sigma1;
model1.sigma2 = sigma2;
model1.M1 = M1;
model1.V1 = V1;


end