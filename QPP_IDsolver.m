function [x,Fval] = QPP_IDsolver(D,num_x,dim_x,x0)
% solve QPP:
% min x'Dx
% s.t. Ax=e,  x>=0
% D=D', and each diagonal group is a diagonal sub-matrix
% A=[1 1 1 0 0 0;
%     0 0 0 1 1 1]; which makes the corresponding block variables as the
%     sum of 1.
% parameters to construct the above problem:
% num_x: the number of block variable vectors
% dim_x: the dimension of each block variable vector, and
% num_x*dim_x==length(diag(D));

% Output: x is a matrix with size dim_x*num_x
% Example:
opt=optimset('display','off');
len=num_x*dim_x;
if len~=size(D,1)
    error('Error: unmatch size in QPP_IDsolver.m');
end
% check B'DB if or not indefinite
B=zeros(len,len-num_x);
for i=1:num_x
    B(1+dim_x*(i-1),1+(dim_x-1)*(i-1):(dim_x-1)*i)=-ones(1,dim_x-1);
    B(2+dim_x*(i-1):dim_x*i,1+(dim_x-1)*(i-1):(dim_x-1)*i)=eye(dim_x-1);
end
evec=eig(B'*D*B);
if isempty(find(evec<0,1))
    A=zeros(num_x,len);
    for i=1:num_x
        A(i,1+(i-1)*dim_x:i*dim_x)=ones(1,dim_x);
    end
   % disp('PD in QPP_IDsolver.m.');
    [x,Fval]=quadprog(D,[],[],[],A,ones(num_x,1),zeros(len,1),[],[],opt);
    return;
end
%disp('ID in QPP_IDsolver.m.');
% if indefinite, solve with initialization
if nargin==4
    [x,Fval]=Solver_x0(D,num_x,dim_x,x0);
    return;
end
% if indefinite, solve with traversal vertex
%disp('    traversal vertex implementation!');
x0=zeros(len,1);
t=dim_x^num_x-1;
for i=1:num_x
    x0((i-1)*dim_x+1,1)=1;
end
[x,Fval]=Solver_x0(D,num_x,dim_x,x0);
for i=1:t
    x0=PlusOne(x0,num_x,dim_x);
    [tx,tFval]=Solver_x0(D,num_x,dim_x,x0);
    if tFval<Fval
        Fval=tFval;
        x=tx;
    end
end

end

function [x0,Fval]=Solver_x0(D,num_x,dim_x,x0)
% must have an inital x0
epsilon=1e-8;
opt=optimset('display','off');
ite=0;
while ite<100
    ite=ite+1;
    x=x0;
    for i=1:num_x
        H=D((i-1)*dim_x+1:i*dim_x,(i-1)*dim_x+1:i*dim_x);
        f=zeros(dim_x,1);
        Row=D((i-1)*dim_x+1:i*dim_x,:);
        for j=1:size(D,2)
            if j<(i-1)*dim_x+1 || j>i*dim_x
                f=f+Row(:,j)*x0(j,1);
            end
        end
        x0((i-1)*dim_x+1:i*dim_x,1)=quadprog(H,f,[],[],ones(1,dim_x),1,zeros(dim_x,1),[],[],opt);                 
    end
    if norm(x-x0)<epsilon
        break;
    end
end  
% compute objective
Fval=x0'*D*x0*0.5;
end

function x=PlusOne(x,num_x,dim_x)
% define x+1 to binary code
% sign denotes if it up to next region
for i=1:num_x
    ind=find(x((i-1)*dim_x+1:i*dim_x,1)>0);
    if i==1
        x(ind,1)=0;
        x(ind+1,1)=x(ind+1,1)+1;
        if ind==dim_x
            x(1,1)=1;
        end
    else
        if length(ind)>1
            x(ind(1)+(i-1)*dim_x,1)=0;
            x(ind(2)+(i-1)*dim_x,1)=0;
            x(ind(2)+(i-1)*dim_x+1,1)=x(ind(2)+(i-1)*dim_x+1,1)+1;
            if ind(2)==dim_x
                x((i-1)*dim_x+1,1)=1;
            end
        elseif x(ind+(i-1)*dim_x,1)>1
            x(ind+(i-1)*dim_x,1)=0;
            x(ind+(i-1)*dim_x+1,1)=1;
        end
    end
end          
end
