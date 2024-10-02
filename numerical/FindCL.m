function [CL,ARL] = FindCL(X,ARL0,ab_sam)

% this function is to find the certain control limit of X distribution
% satisfied the target ARL0
% if there is NaN in X (occuring in t5 and g3 distribution),find it and set
% it to be inf based on Hawkins 2008 (meaning the normalized U has inf number)
if nargin == 2
    ab_sam = 1;
end


index=isnan(X);
X(index)=inf; % then set it to be inf, signing an out-of-control signal
CLMax = max(max(X));
ARLMax = ComputeARL(X,CLMax,ab_sam);
if ARLMax < ARL0
    warning(['The maximum CL and ARL are ',num2str(CLMax),' and ',...
        num2str(ARLMax),'.']);
    CL = CLMax;
    ARL = ARLMax;
    return;
else 
    options = statset('TolX',eps);
    CL = fminbnd(@(x) abs(ComputeARL(X,x,ab_sam)-ARL0),0,CLMax);
    if nargout==2
        ARL = ComputeARL(X,CL,ab_sam);
    end
end
% ComputeARL(X,10,changeP)
end

function ARL = ComputeARL(Test,UCL,ab_sam)
% [u,v] = find(TF);
% ARL = mean(u(diff([0;v])==1));
[rep,T_sum] = size(Test);
detectP = ones(1,rep);
profile_start = ab_sam;
for s = 1:rep
    temp = find(Test(s,:)>UCL, 1 );
    if isempty(temp)
        detectP(s) = T_sum;
    else
        detectP(s) = temp;
    end
end

ARL = mean(detectP)-profile_start+1;

end