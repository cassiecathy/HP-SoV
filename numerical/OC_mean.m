function ocY = OC_mean(icY,ab_sensor,ab_stage,ab_sam,shift)
[~,~,nsam] = size(icY{1});
ocY = icY;

for i = ab_sam:nsam
    for s = ab_stage
        for j = ab_sensor
            ocY{s}(:,j,i) = icY{s}(:,j,i).*(1+shift);
        end
    end
end

end
