function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
    trainK=expX2Kernel(trainD,trainD,gamma);
    testK=expX2Kernel(testD,trainD,gamma);
end
function [Kernel]=expX2Kernel(X,Y,gamma)
    n=size(X,2);
    m=size(Y,2);
    Kernel=zeros(n,m);
    for i=1:n
        for j=1:m
            numer=(X(:,i)-Y(:,j)).^2;
            denom=X(:,i)+Y(:,j)+eps;
            Kernel(i,j)=sum(numer./denom);
        end
    end
    Kernel=exp(-1*Kernel/gamma);
end