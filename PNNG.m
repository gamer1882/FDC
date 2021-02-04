function Y= PairNNG(X,M,C,k)
% PairNNG initialize the input X with pairwise constraint M and C to get Y.
% X: each row is a sample
% Y: the initial cluster label.
% M: must-link set with pairs as (1,2), means 1st and 2nd samples in a
%     cluster
% C: cannot-link set with pairs as (2,3), means 2nd and 3rd samples not in
%     a cluster
% k: the number of clusters you want.
% more details see Semi-supervised fuzzy clustering with fuzzy pairwise
% constraints, writed by Zhen Wang, wangzhen@imu.edu.cn,2020.2.22.
m=size(X,1);
m1=size(M,1);
m2=size(C,1);
% put the samples of M forward in tX, and the samples of both M and C next
%     in tX, and the samples of C next, and the rest to fast search C 
ind1=unique(reshape(M,2*m1,1));
len_ind1=length(ind1);
ind2=unique(reshape(C,2*m2,1));
len_ind2=length(ind2);
indboth=intersect(ind1,ind2);
indout=setdiff(1:m,ind1);
indout=setdiff(indout,ind2);
indout=indout';
len_b=length(indboth);  % length of indboth
if len_b>0
    ind1_2=setdiff(ind1,indboth);
    ind2_1=setdiff(ind2,indboth);
else
    ind1_2=ind1;
    ind2_1=ind2;
end
tX=X([ind1_2;indboth;ind2_1;indout],:);
ind1=[ind1_2;indboth]; %按顺序重新组织
ind2=[indboth;ind2_1];
tM=zeros(m1,2);
tC=zeros(m2,2);
for i=1:m1
    tM(i,1)=find(ind1==M(i,1),1);
    tM(i,2)=find(ind1==M(i,2),1);
end
for i=1:m2
    tC(i,1)=length(ind1_2)+find(ind2==C(i,1),1);
    tC(i,2)=length(ind1_2)+find(ind2==C(i,2),1);
end
% **following only use tX, tY with tM and tC**
tY=zeros(m,1);
book=-ones(m,m); % distance matrix
% set special distances in book by M and C
for i=1:m1
    book(tM(i,1),tM(i,2))=0;
    book(tM(i,2),tM(i,1))=0;
end
for i=1:m2
    book(tC(i,1),tC(i,2))=inf;
    book(tC(i,2),tC(i,1))=inf;
end
cnt=zeros(m,m); % connection matrix, 1 denotes link and 0 not
%adv1=cell(m1,1); % advocacy matrix for each pair in M,
adv2=-ones(m2,2); % advocacy matrix for each pair in C, 
                %    example: adv{1,1}=[1,3,4] means the 1st, 3rd, and 4th
                %        clusters are the advocacy cluster w.r.t. 1st pair.                             
% compute the distance matrix
mD=m*(m-1)/2;
D=zeros(mD,1); % used for 1(b)
pD=zeros(mD,2);% used for 1(b)
tmp=1;
for i=1:m
    for j=i+1:m
        if book(i,j)<0
            val=norm(tX(i,:)-tX(j,:));
            book(i,j)=val;
            book(j,i)=book(i,j);
            D(tmp,1)=val;
            pD(tmp,1)=i;
            pD(tmp,2)=j;
            tmp=tmp+1;  
        end
    end
end
len_D=tmp-1;
% 1.(a) connect pair in M and label them
lab=1;
for i=1:m1
    if tY(tM(i,1))<1 && tY(tM(i,2))<1
        tY(tM(i,1))=lab;
        tY(tM(i,2))=lab;
        lab=lab+1;        
    elseif tY(tM(i,1))<1
        tY(tM(i,1))=tY(tM(i,2));
    elseif tY(tM(i,2))<1
        tY(tM(i,2))=tY(tM(i,1));
    elseif tY(tM(i,1))~=tY(tM(i,2))                  
        allind=GetLinkInd(cnt,tM(i,1));   
        tY(allind)=tY(tM(i,2));
    end
    cnt(tM(i,1),tM(i,2))=1;
    cnt(tM(i,2),tM(i,1))=1;
end
% if both in M and C, add advocacy in C for the both
for i=len_ind1-len_b+1:len_ind1
    ind_both_one=IndInC(tC,i);
    for j=1:length(ind_both_one)
        if isempty(find(adv2(ind_both_one(j),:)==tY(i),1))
            adv2(ind_both_one(j),1)=tY(i);
        end
    end
end
% 1.(b) sort the NN distances D and pD
for i=1:len_D-1
    tmp=i;
    for j=i+1:len_D
        if D(tmp)>D(j)
            tmp=j;
        end
    end
    if tmp~=i
        tmpD=D(i); D(i)=D(tmp); D(tmp)=tmpD;
        tmpPD=pD(i,:); pD(i,:)=pD(tmp,:); pD(tmp,:)=tmpPD;
    end
end
% 1.(c) combine the pair in pD
for i=1:len_D
    if tY(pD(i,1))<1 && tY(pD(i,2))<1 % both have no label
        if (pD(i,1)<=len_ind1-len_b || pD(i,1)>len_ind1+len_ind2-len_b) && (pD(i,2)<=len_ind1-len_b || pD(i,2)>len_ind1+len_ind2-len_b)
            cnt(pD(i,1),pD(i,2))=1;
            cnt(pD(i,2),pD(i,1))=1;
            tY(pD(i,1))=lab;
            tY(pD(i,2))=lab;
            lab=lab+1;            
        else
            ind_pD1=[];
            ind_pD2=[];
            if pD(i,1)>len_ind1-len_b && pD(i,1)<=len_ind1+len_ind2-len_b
                ind_pD1=IndInC(tC,pD(i,1)); % 输出后参数在前参数中所存在的行号，可能多个，不会在一对中同时出现
            end
            if pD(i,2)>len_ind1-len_b && pD(i,2)<=len_ind1+len_ind2-len_b
                ind_pD2=IndInC(tC,pD(i,1));
            end
            if isempty(intersect(ind_pD1,ind_pD2))
                cnt(pD(i,1),pD(i,2))=1;
                cnt(pD(i,2),pD(i,1))=1;
                tY(pD(i,1))=lab;
                tY(pD(i,2))=lab;
                if ~isempty(ind_pD1)
                    for j=1:length(ind_pD1)
                        tmpind_adv=find(adv2(ind_pD1(j),:)<0,1);
                        adv2(ind_pD1(j),tmpind_adv)=lab;
                    end
                end
                if ~isempty(ind_pD2)
                    for j=1:length(ind_pD2)
                        tmpind_adv=find(adv2(ind_pD2(j),:)<0,1);
                        adv2(ind_pD2(j),tmpind_adv)=lab;
                    end
                end
                lab=lab+1;
            end
        end
    elseif tY(pD(i,1))>0 && tY(pD(i,2))<1        % one have label and the other not
        if pD(i,2)<=len_ind1-len_b || pD(i,2)>len_ind1+len_ind2-len_b % the unlabel one is out of C
            if cnt(pD(i,1),pD(i,2))~=1
                cnt(pD(i,1),pD(i,2))=1;
                cnt(pD(i,2),pD(i,1))=1;
                tY(pD(i,2))=tY(pD(i,1));
            end
        else
            ind_cluster_adv=IndInAdvC(adv2,tY(pD(i,1)));
            ind_sample_adv=IndInC(tC,pD(i,2));
            if isempty(ind_cluster_adv) || isempty(intersect(ind_cluster_adv,ind_sample_adv))
                if cnt(pD(i,1),pD(i,2))~=1
                    cnt(pD(i,1),pD(i,2))=1;
                    cnt(pD(i,2),pD(i,1))=1;
                    tY(pD(i,2))=tY(pD(i,1));
                end
                for j=1:length(ind_sample_adv)
                    tmpind_adv=find(adv2(ind_sample_adv(j),:)<0,1);
                    adv2(ind_sample_adv(j),tmpind_adv)=tY(pD(i,1));
                end
            end
        end
    elseif tY(pD(i,2))>0 && tY(pD(i,1))<1        % one have label and the other not
        if pD(i,1)<=len_ind1-len_b || pD(i,1)>len_ind1+len_ind2-len_b
            if cnt(pD(i,1),pD(i,2))~=1
                cnt(pD(i,1),pD(i,2))=1;
                cnt(pD(i,2),pD(i,1))=1;
                tY(pD(i,1))=tY(pD(i,2));
            end
        else
            ind_cluster_adv=IndInAdvC(adv2,tY(pD(i,2))); % cluster advocacy
            ind_sample_adv=IndInC(tC,pD(i,1));  %
            if isempty(ind_cluster_adv) || isempty(intersect(ind_cluster_adv,ind_sample_adv))
                if cnt(pD(i,1),pD(i,2))~=1
                    cnt(pD(i,1),pD(i,2))=1;
                    cnt(pD(i,2),pD(i,1))=1;
                    tY(pD(i,1))=tY(pD(i,2));
                end
                for j=1:length(ind_sample_adv)
                    tmpind_adv=find(adv2(ind_sample_adv(j),:)<0,1);
                    adv2(ind_sample_adv(j),tmpind_adv)=tY(pD(i,2));
                end
            end
        end
    elseif tY(pD(i,2))~=tY(pD(i,1))  % both have labels which not equal
        ind_cluster_adv1=IndInAdvC(adv2,tY(pD(i,1)));
        ind_cluster_adv2=IndInAdvC(adv2,tY(pD(i,2)));
        if isempty(ind_cluster_adv1) && isempty(ind_cluster_adv2) % clusters have no advocacy
            ind_sample_link1=GetLinkInd(cnt,pD(i,1));
            if cnt(pD(i,1),pD(i,2))~=1
                cnt(pD(i,1),pD(i,2))=1;
                cnt(pD(i,2),pD(i,1))=1;
                tY(ind_sample_link1)=tY(pD(i,2));
            end
        elseif isempty(ind_cluster_adv1)  % 1st cluster has no advocacy
            ind_sample_link1=GetLinkInd(cnt,pD(i,1));
            if cnt(pD(i,1),pD(i,2))~=1
                cnt(pD(i,1),pD(i,2))=1;
                cnt(pD(i,2),pD(i,1))=1;
                tY(ind_sample_link1)=tY(pD(i,2));
            end
        elseif isempty(ind_cluster_adv2) % 2nd cluster has no advocacy 
            ind_sample_link1=GetLinkInd(cnt,pD(i,2));
            if cnt(pD(i,1),pD(i,2))~=1
                cnt(pD(i,1),pD(i,2))=1;
                cnt(pD(i,2),pD(i,1))=1;
                tY(ind_sample_link1)=tY(pD(i,1));
            end
        else    % two clusters have advocacy
            ind_cluster_adv12=intersect(ind_cluster_adv2,ind_cluster_adv1);
            if isempty(ind_cluster_adv12) % the advocacies are different
                ind_sample_link1=GetLinkInd(cnt,pD(i,2));
                if cnt(pD(i,1),pD(i,2))~=1
                    cnt(pD(i,1),pD(i,2))=1;
                    cnt(pD(i,2),pD(i,1))=1;
                    tY(ind_sample_link1)=tY(pD(i,1));
                end
                for j=1:length(ind_cluster_adv2)                    
                    adv2(ind_cluster_adv2(j),adv2(ind_cluster_adv2(j),:)==tY(pD(i,2)))=tY(pD(i,1));
                end                
            end
        end
    end
end
% addition for noises, each sample is a cluster 
if ~isempty(find(tY<1,1))
    %disp('Warning in PairNNG.m: there are samples with no label, and results may be unstable!');
    tY=TansferZeroToEach(tY);
end
lab_cur=unique(tY);
k_cur=length(lab_cur);
% 2 and 3  %连接时可以用先判断是否已连来节省计算？
if k_cur>k
    len_lab=length(lab_cur);
    len_dis_pair=len_lab*(len_lab-1)/2;
    dis_pair=zeros(len_dis_pair,1);  % pair of clusters' distances
    ind_pair=zeros(len_dis_pair,2);
    tmp=1;
    for i=1:len_lab
        for j=i+1:len_lab
            ind_lab1=find(tY==lab_cur(i));
            ind_lab2=find(tY==lab_cur(j));
            dis_pair(tmp,1)=Housdorff(book,ind_lab1,ind_lab2);
            ind_pair(tmp,1)=i;
            ind_pair(tmp,2)=j;
            tmp=tmp+1;
        end
    end
    for i=1:len_dis_pair-1  % sort dis_pair and ind_pair ascend
        tmp=i;
        for j=i+1:len_dis_pair
            if dis_pair(tmp)>dis_pair(j)
                tmp=j;
            end
        end
        if tmp~=i
            tmpval=dis_pair(i); dis_pair(i)=dis_pair(tmp); dis_pair(tmp)=tmpval;
            tmppair=ind_pair(i,:);ind_pair(i,:)=ind_pair(tmp,:);ind_pair(tmp,:)=tmppair;
        end
    end
    for i=1:len_dis_pair
        if lab_cur(ind_pair(i,1))~=lab_cur(ind_pair(i,2))
            ind_adv1=IndInAdvC(adv2,lab_cur(ind_pair(i,1))); % adv ind for the 1st cluster
            ind_adv2=IndInAdvC(adv2,lab_cur(ind_pair(i,2))); % adv ind for the 2st cluster
            if isempty(intersect(ind_adv1,ind_adv2))
                tY(tY==lab_cur(ind_pair(i,2)))=lab_cur(ind_pair(i,1)); % change all labels
                lab_cur(ind_pair(i,2))=lab_cur(ind_pair(i,1)); % change the current label list
                for j=1:length(ind_adv2)
                    adv2(ind_adv2(j),adv2(ind_adv2(j),1)==lab_cur(ind_pair(i,2)))=lab_cur(ind_pair(i,1));
                end
                k_cur=k_cur+1;
                if k_cur==k
                    break;
                end
            end
        end
    end
elseif k_cur<k
    dis_pair=zeros(mD,1);
    ind_pair=zeros(mD,2);
    tmpk=1;
    for i=1:m
        for j=i+1:m
            if cnt(i,j)~=0 && book(i,j)~=0
                dis_pair(tmpk,1)=book(i,j);
                ind_pair(tmpk,1)=i;
                ind_pair(tmpk,2)=j;
                tmpk=tmpk+1;
            end
        end
    end
    tmpk=tmpk-1; % before this the value meaningful
    % sort the distances of pair samples descend
    for i=1:tmpk-1
        tmp=i;
        for j=i+1:tmpk
            if dis_pair(tmp)<dis_pair(j)
                tmp=j;
            end
        end
        if tmp~=i
            tmpval=dis_pair(i); dis_pair(i)=dis_pair(tmp); dis_pair(tmp)=tmpval;
            tmppair=ind_pair(i,:);ind_pair(i,:)=ind_pair(tmp,:);ind_pair(tmp,:)=tmppair;
        end
    end
    % disconnect a pair in sequence, don't need to consider the pair in M because of distance 0
    for i=1:tmpk
        cnt(ind_pair(i,1),ind_pair(i,2))=0;
        cnt(ind_pair(i,2),ind_pair(i,1))=0;
        ind_conn=GetLinkInd(cnt,ind_pair(i,1));
        if isempty(find(ind_conn==ind_pair(i,2),1))
            tY(ind_conn,1)=lab;
            lab=lab+1;
            k_cur=k_cur+1;
            if k_cur==k
                break;
            end
        end
    end
end
tY=CorrectLabel(tY);
Y=zeros(m,1);
Y(ind1,1)=tY(1:len_ind1,1);
Y(ind2,1)=tY(len_ind1-len_b+1:len_ind1+len_ind2-len_b,1);
Y(indout,1)=tY(len_ind1+len_ind2-len_b+1:m,1);
end


function ind=IndPairInC(tC,i)
% find the corresponding pair sample of i in tC
ind=[];
for j=1:size(tC,1)
    if i==tC(j,1)
        ind=[ind;tC(j,2)];
    elseif i==tC(j,2)
        ind=[ind;tC(j,1)];
    end
end
end

function ind=IndInC(tC,i)
% find the rows of i in tC
ind=[];
for j=1:size(tC,1)
    if i==tC(j,1) || i==tC(j,2)
        ind=[ind;j];
    end
end
end

function ind=IndInAdvC(adv,k)
% find the row of k-th cluster in adv2 of tC 
ind=[];
for i=1:size(adv,1)
    if ~isempty(find(adv(i,:)==k,1))
        ind=[ind;i];
    end
end
end


function ind=GetLinkInd(cnt,i)
% return the indexes that linked the i-th node
ind=i;
new=[];
diff=i;
while 1
    for j=1:length(diff)
        newid=find(cnt(:,diff(j))>0);
        new=[new;newid];
    end
    diff=setdiff(new,ind);
    if isempty(diff)
        return;
    end
    ind=[ind;diff];
    new=[];
end      
end

function val=Housdorff(book,v1,v2)
% hausdorff distance
% v1 is the index of 1-st set
% v2 is the index of 2-nd set
m1=length(v1);m2=length(v2);
val2=-inf;
for i=1:m1
    val1=inf;
    for j=1:m2
        if val1>book(v1(i),v2(j))
            val1=book(v1(i),v2(j));
        end
    end
    if val2<val1
        val2=val1;
    end
end
val3=-inf;
for i=1:m1
    val1=inf;
    for j=1:m2
        if val1>book(v1(i),v2(j))
            val1=book(v1(i),v2(j));
        end
    end
    if val3<val1
        val3=val1;
    end
end
val=max(val2,val3);
end

function tY=TansferZeroToEach(Y)
% 0 in tY is transferred to a unique label
ind=find(Y<1);
tY=Y;
if ~isempty(ind)
    label=max(Y)+1;
    for i=1:length(ind)
        tY(ind(i),1)=label;
        label=label+1;
    end
end
        
end