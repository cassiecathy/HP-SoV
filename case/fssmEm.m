function [model, llh] = fssmEm(Y,User_options)
% Input:
% Y: cell(1,nstage):ntime*nsensor*nsam

% Output:
%   model: trained model structure
%   llh: loglikelihood

%% Input parameters checking
nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),~] = size(Y{s});
end

% The default value 
options = struct('lambda',0,'prop',0.9,'norm',0);
% Input the user setting
if nargin == 2
    if isfield(User_options, 'lambda')
        options.lambda = User_options.lambda;
    end
    if isfield(User_options, 'npc')
        options.npc = User_options.npc;
    elseif isfield(User_options, 'prop')
        options.prop = User_options.prop;
    end    
end

%% initialization and EM
model = init(Y,options);

tol = 5e-4;
maxIter = 200;
llh = zeros(1,maxIter);
for iter = 1:maxIter
    [Ex, Exx, Exy, llh(iter+1)] = Estep(model,Y); % E step
    if abs(llh(iter+1)-llh(iter)) < tol*abs(llh(iter))
        fprintf('The interation number in the EM loop is %4.2f. \n',iter);
        break; 
    end   % check likelihood for convergence
    model = Mstep(Y, Ex, Exx, Exy,model,options); % M step
end
llh = llh(2:iter+1);
end

