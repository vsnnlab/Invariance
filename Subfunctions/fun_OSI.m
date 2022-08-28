function [d] = fun_OSI(rep)

d = zeros(size(rep,1),1);
for ii = 1:size(d,1)
    [~,indOrder] = sort(nanmean(rep(ii,:,:),3),'descend');
    d(ii) = (nanmean(rep(ii,1,:),3)-nanmean(rep(ii,indOrder(2),:),3))./sqrt((nanstd(rep(ii,1,:),[],3).^2+nanstd(rep(ii,indOrder(2),:),[],3).^2)./2);
end
end