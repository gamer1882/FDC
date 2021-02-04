function tY=CorrectLabel(Y)
% Correct label from 1, 2, 3, ... , to max
labels=unique(Y);
m=size(Y,1);
tY=zeros(m,1);
for i=1:length(labels)
    tY(Y==labels(i),1)=i;
end
end