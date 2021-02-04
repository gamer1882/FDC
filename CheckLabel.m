function cY=CheckLabel(Y)
% revise labels if they are not from 1,2,3,...,k
% Y is a column vector
if isempty(Y) || size(Y,2)>1
    cY=[];
    disp('Error: empty Y in ResClustering.m');
    return;
end
uY=unique(Y);
cY=Y;
k=1;
for i=1:length(uY)
    cY(Y==uY(i),1)=k;
    k=k+1;
end
end