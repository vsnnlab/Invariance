function [array_idx_RowCol] = fun_backtracking_C4toC3_XY(net_rand,ind)

%% conv5 face neuron index
net = net_rand;
ind_netLayer = [2, 6, 10, 12, 14];
ind_layer = 4;
Weight = net.Layers(ind_netLayer(ind_layer)).Weights;
    
array_sz = [55 55 96; 13 13 256; 13 13 384; 13 13 384; 13 13 256];
ind_all_XY = ind;
array_idx_RowCol = zeros(size(Weight,1)*size(Weight,2),2);


for iii = 1:length(ind_all_XY)
%     disp(['I_step : ',num2str(iii)])
    ind_layer = 4;
    ind_top = iii;
    [row, col, chan] = ind2sub(array_sz(ind_layer,:),ind_all_XY(ind_top)); % top 1
    
    %% network
    % relu4 -> conv 4
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % relu4 -> conv4 -> relu3
    w = size(Weight(:,:,:,chan),1); % filter size
    s = net.Layers(ind_netLayer(ind_layer)).Stride;
    p = net.Layers(ind_netLayer(ind_layer)).PaddingSize;
    
    row_pre = (row-1)*s(1)+w-2*p(1);
    col_pre = (col-1)*s(1)+w-2*p(1);
    [cc,rr] = meshgrid(-(w(1)-1)/2:1:(w(1)-1)/2);
    row_new = rr+row_pre; col_new = cc+col_pre;
    row_new = row_new(:); col_new = col_new(:);
    array_idx_RowCol(:,1) = row_new; array_idx_RowCol(:,2) = col_new;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

end