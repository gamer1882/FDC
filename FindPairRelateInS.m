function group_pair=FindPairRelateInS(pairS,labels)
% used in SFC_core_iY.m
% find the group_pair that has labels
used_pair=zeros(1,size(pairS,1));
used_lab=zeros(1,length(labels));
first=find(used_lab==0,1);
group_pair={};
n_group=1;
while ~isempty(first)
    pairs=[];
    ind=first; % q
    sign=1;
    while sign==1  
        sign=0;
        for i=1:length(ind)
            if used_lab(ind(i))~=1
                [row,col]=find(pairS==labels(ind(i)));
                for j=1:length(row)
                    if used_pair(row(j))==0
                        used_pair(row(j))=1;
                        pairs=[pairs;pairS(row(j),:)];                        
                    end
                end            
                used_lab(ind(i))=1;          
                for j=1:length(col)
                    ind_one=find(labels==pairS(row(j),3-col(j)),1);
                    if used_lab(ind_one)==0 && isempty(find(ind==ind_one,1))
                        ind=[ind;ind_one];
                        sign=1;
                    end
                end
            end
        end        
    end
    first=find(used_lab==0,1);
    group_pair{n_group}=pairs;
    n_group=n_group+1;
end
end