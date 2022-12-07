%% ��ԭ�������������������ں����仯��Χ����������ܼ��������
clear;
load('beta_zero.mat');
X=[a1.data,a2.data,...
        a5.data,x1.data,...
        x2.data,x3.data,...
        x4.data,x5.data];    
rho=.15;

% ǰ3000deȨֵ,ÿ30����
for i=1:100
    Y=X(1+30*(i-1):30*i,:);
    %center
    c=Y(1:floor(size(Y,1)/14):size(Y,1),:);
    for j=1:size(c,1)
        x=0;
        for k=1:size(Y,1)
            a(k)=norm(c(j,:)-Y(k,:));
        end
        for kk=1:15
            [xx,I]=min(a);
            x=x+xx^2;
            a(I)=[];
        end
        dist(j)=x^.5/15;
    end
    %bias
    sigma=1./(sqrt(2)*dist);
    
    %radial basis function
    for j=1:size(Y,1)
        for k=1:size(c,1)
            h(k,j)=exp(-(Y(j,:)-c(k,:))*(Y(j,:)-c(k,:))'*(sigma(k)^2));
        end
    end
    h=[h;ones(1,size(h,2))];
    w(:,i)=(h*h'+rho*ones(size(h,1)))^(-1)*h*Fo.data(1+30*(i-1):30*i);
    clear h
end

%����ʣ���Ȩֵ���
Y=X(3001:end,:);
c=Y(1:floor(size(Y,1)/14):size(Y,1),:);
for j=1:size(c,1)
    x=0;
    for k=1:size(Y,1)
        a(k)=norm(c(j,:)-Y(k,:));
    end
    for kk=1:1000
        [xx,I]=min(a);
        x=x+xx^2;
        a(I)=[];
    end
    dist(j)=x^.5/1000;
end
%bias
sigma=1./(sqrt(2)*dist);

%radial basis function
for j=1:size(Y,1)
    for k=1:size(c,1)
        h(k,j)=exp(-(Y(j,:)-c(k,:))*(Y(j,:)-c(k,:))'*(sigma(k)^2));
    end
end
h=[h;ones(1,size(h,2))];
v=(h*h'+rho*ones(size(h,1)))^(-1)*h*Fo.data(3001:end);
