function [ocY,icY0] = OCgen(model0,nIC,nOC,ab_senario,shift,ab_sensor,ab_stage,ab_sam,isshow)
if nargin == 8
    isshow = 0;
end
[icY,~] = genfssm(model0,nOC);  
icY0 = genfssm(model0,nIC);  
% options.X_gen = X_gen;
% options.norm = 1;

if ~(shift == 0)
    switch ab_senario
        case 1
            ocY = OC_mean(icY,ab_sensor,ab_stage,ab_sam,shift);
        case 2
            ocY = OC_slope(icY,ab_sensor,ab_stage,ab_sam,shift); 
        case 3
            ocY = OC_mix(icY,ab_sensor,ab_stage,ab_sam,shift); 
    end
else
    ocY = icY;
end

if isshow
    temp = 1;
    nstage = size(icY,2);
    Ysum = cell(1,nstage);
    for s = 1:nstage
        Ysum{s}(:,:,1) = icY{s}(:,:,temp);
        Ysum{s}(:,:,2) = ocY{s}(:,:,temp);
    end
    plotY(Ysum,1,{'IC','OC'});
end

            
end








