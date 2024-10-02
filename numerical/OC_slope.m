function ocY = OC_slope(icY,ab_sensor,ab_stage,ab_sam,shift)
[~,~,nsam] = size(icY{1});
ocY = icY;

for i = ab_sam:nsam
    for s = ab_stage
        for j = ab_sensor
            for t = 1:length(ocY{s}(:,j,i))
            ocY{s}(t,j,i) = icY{s}(t,j,i)-shift*t;
            end
        end
    end
end

end
