function [pY,u]= SFC_core_iY(X,iY,S,c1,c2)
% Frame of semi-supervised fuzzy clustering with fuzzy pairwise constraints
% iY is the initial labels and has been corrected by CorrectLabels()
% c1 is beta and c2 is alpha
% S: a triad -- (i,j,s) the i-th and j-th samples with -1<=s<=1 for fuzzy
%     pair
eps=1e-5;
k=max(iY);
[m,n]=size(X);
u=zeros(m,k); % a row is a sample, and a column is a cluster
delta=zeros(k,n); % a row is a cluster's center
if ~isempty(S)
    pairS=S(:,[1 2]);
    ind_unique_S=unique(pairS);
    group_pair=FindPairRelateInS(pairS,ind_unique_S);
    labels_group=cell(1,size(group_pair,2));
    group_u0=cell(1,size(group_pair,2));
    for i=1:size(group_pair,2)
        labels_group{1,i}=unique(group_pair{1,i});
        group_u0{1,i}=zeros(length(labels_group{1,i})*k,1);%%%
    end
    ind_out_S=setdiff(1:m,ind_unique_S);
else
    ind_out_S=1:m;
end
% initial u
for i=1:m
    u(i,iY(i))=1;
end
% begin iteration
newk=k;
num_del_cluster=[]; % if delete a cluster, then the loop restart
ite=0;
while ite<100
    ite=ite+1;
    old_delta=delta;
    % compute delta
    for j=1:newk
        dec_vec=u(:,j).^2-c2;
        if sum(dec_vec)>eps
            val=0;
            for i=1:m
                val=val+dec_vec(i)*X(i,:);
            end
            delta(j,:)=val/sum(dec_vec);
        else
            num_del_cluster=[num_del_cluster;j];
        end
    end
    if ~isempty(num_del_cluster)
        if size(u,2)==1
            break;
        else
            u(:,num_del_cluster)=[];
            delta(num_del_cluster,:)=[];
            newk=size(u,2);
            if newk==0
                pY=ones(m,1);
                return;
            end
        end
    else
        term=0;
        for i=1:newk
            term=term+norm(delta(i,:)-old_delta(i,:))^2;
        end
        if term<eps
            break;
        end
    end
    % compute u for case 1
    for i=1:length(ind_out_S)
        d=0;
        for j=1:newk
            dj=norm(X(ind_out_S(i),:)-delta(j,:))^2;
            if dj<eps
                u(ind_out_S(i),:)=0;
                u(ind_out_S(i),j)=1;
                break;
            else
                u(ind_out_S(i),j)=1/dj;
                d=d+u(ind_out_S(i),j);
            end
        end
        if d~=0
            u(ind_out_S(i),:)=u(ind_out_S(i),:)/d;
        end
    end
    % compute u for case 2
    if ~isempty(S)
        for i=1:size(group_pair,2)
            DforQPP=zeros(length(labels_group{1,i})*newk,length(labels_group{1,i})*newk);
            tmp=1;
            for j1=1:length(labels_group{1,i})
                for j2=1:newk
                    DforQPP(tmp,tmp)=norm(X(labels_group{1,i}(j1),:)-delta(j2,:))^2;
                    tmp=tmp+1;
                end
            end
            for j=1:size(group_pair{1,i},1)
                s_thispair=GetOriginS(S,group_pair{1,i}(j,:));
                if s_thispair>0
                    ind_thispair=find(labels_group{1,i}==group_pair{1,i}(j,1),1);
                    ind_thispair2=find(labels_group{1,i}==group_pair{1,i}(j,2),1);
                    val_thispair=c1/2*s_thispair;
                    for j2=1:newk
                        DforQPP((ind_thispair-1)*newk+j2,(ind_thispair-1)*newk+j2)=DforQPP((ind_thispair-1)*newk+j2,(ind_thispair-1)*newk+j2)+val_thispair;
                        DforQPP((ind_thispair2-1)*newk+j2,(ind_thispair2-1)*newk+j2)=DforQPP((ind_thispair2-1)*newk+j2,(ind_thispair2-1)*newk+j2)+val_thispair;
                    end
                end
                coef=-c1*m/size(S,1)*s_thispair; % coef of ui and uj in a pair
                ind1=find(labels_group{1,i}==group_pair{1,i}(j,1),1);
                ind2=find(labels_group{1,i}==group_pair{1,i}(j,2),1);
                row_ind=(ind1-1)*newk+1:ind1*newk;
                col_ind=(ind2-1)*newk+1:ind2*newk;
                for r=1:length(row_ind)
                    DforQPP(row_ind(r),col_ind(r))=coef;
                    DforQPP(col_ind(r),row_ind(r))=coef;
                end
            end
            if length(labels_group{1,i})<=3 && newk <=3
                group_u0{1,i} = QPP_IDsolver(DforQPP,length(labels_group{1,i}),newk);
            else
                if ~isempty(num_del_cluster)
                    group_u0{1,i}=zeros(length(labels_group{1,i})*newk,1);
                    group_u0{1,i}=QPP_IDsolver(DforQPP,length(labels_group{1,i}),newk,group_u0{1,i});
                else
                    group_u0{1,i}=QPP_IDsolver(DforQPP,length(labels_group{1,i}),newk,group_u0{1,i});
                end
            end
            for j=1:length(labels_group{1,i})
                u(labels_group{1,i}(j),:)=group_u0{1,i}((j-1)*newk+1:j*newk)';
            end
        end
    end
    num_del_cluster=[];
end
[~,pY]=max(u,[],2);
pY=CorrectLabel(pY);
end

function val=GetOriginS(S,pair)
% get the 3-rd value for the pair from S
for i=1:size(S,1)
    if S(i,1)==pair(1) && S(i,2)==pair(2)
        val=S(i,3);
        return;
    end
end
end


