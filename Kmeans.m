function []=Question2_5_1()
    X = load('/Users/chaitrahegde/Documents/ML CSE 512/Lecture Slides/Homework5/hw5data/digit/digit.txt');
    Y = load('/Users/chaitrahegde/Documents/ML CSE 512/Lecture Slides/Homework5/hw5data/digit/labels.txt');
    k=[2 4 6];
    for i=1:size(k,2)
        [clusters,centroids,iteration,SS,totalSS]=Kmean(k(i),X);
        [p1,p2,p3]=paircounting(Y,clusters);
        disp("K= "+k(i)+ " iterations= "+iteration+ " group sum of squares= "+totalSS+ " p1= "+p1+ " p2= "+p2+ " p3= "+p3);
    end

end 

function [clusters,centroids,iteration,SS,totalSS]=Kmean(k,X)
    centers=X(1:k,:);
    n=size(X,1);
    columns=size(X,2);
    clusters=zeros(n,1);
    dist=zeros(k,1);
    stop=0;
    iteration=0;
    while(stop==0 && iteration<20)
        clusterold=clusters;
        for i=1:n
            for j=1:k
                dist(j,:)=norm(X(i,:)-centers(j,:));
            end
            [~, ind] = min(dist);
            clusters(i,:)=ind;
        end
        for j=1:k
            centers(j,:) = mean(X(clusters==j,:),1);
        end
        
        centroids=centers;
        if(clusterold==clusters)
            stop=1;
        end
        iteration=iteration+1;
    end
    [SS,totalSS]=findSS(X,clusters,centroids,k);

end 

function [SS,totalSS]=findSS(X,clusters,centroids,k)
    for i=1:k
        values=X(clusters==i,:);
        SS(i,:)=(sum(norm(X(clusters==i,:)-centroids(i,:)))^2);
    end
    totalSS=sum(SS);
end

function [p1,p2,p3]=paircounting(Y,clusters)
    n=size(Y,1);
    countp1=0;
    countp2=0;
    allp1pairs=0;
    allp2pairs=0;
    for i=1:n
        for j=i+1:n
            if(Y(i,:)==Y(j,:))
                allp1pairs=allp1pairs+1;
                if(clusters(i,:)==clusters(j,:))
                    countp1=countp1+1;
                end
            elseif (Y(i,:)~=Y(j,:))
                allp2pairs=allp2pairs+1;
                if(clusters(i,:)~=clusters(j,:))
                    countp2=countp2+1;
                end
            end
        end
    end
    p1=countp1/allp1pairs;
    p2=countp2/allp2pairs;
    p3=(p1+p2)/2;
end