function out = blkdiag_lpy(B)
nstage = size(B,2);

out = [];
for i = 1:nstage
    out = blkdiag(out,B{i});
end

end