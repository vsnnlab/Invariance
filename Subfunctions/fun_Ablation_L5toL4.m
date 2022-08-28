function [net_ablation] = fun_Ablation_L5toL4(cell_list_L5,array_sz,net_rand,W_conv5, cell_list_L4, reverse)
indLayer = 5;
array_sz_conv5 = size(W_conv5);
[c,r] = meshgrid(1:size(W_conv5,1),1:size(W_conv5,2));
IND_rc = [r(:), c(:)];

W_conv5_ablation = W_conv5;

for ii = 1:length(cell_list_L5)
    tmp_IND_rc = IND_rc;
    ind_C5_cell = cell_list_L5(ii);
    [~,~,IND_C5_chan] = ind2sub(array_sz(indLayer,:),ind_C5_cell);
    t_w = W_conv5(:,:,:,IND_C5_chan);
    t_w = t_w(:);
    
    [IND_C4_rc] = fun_backtracking_C5toC4_XY(net_rand,ind_C5_cell);
    
    i1 = find((IND_C4_rc(:,1)<=0|IND_C4_rc(:,1)>array_sz(indLayer-1,1)));
    if length(i1)>0
        IND_C4_rc(i1,:) = []; tmp_IND_rc(i1,:) = [];
    end
    i2 = find((IND_C4_rc(:,2)<=0|IND_C4_rc(:,2)>array_sz(indLayer-1,1)));
    if length(i2)>0
        IND_C4_rc(i2,:) = []; tmp_IND_rc(i2,:) = [];
    end
    
    if IND_C5_chan <= array_sz(indLayer,3)/2
        tmp_IND_C4_row = repmat(IND_C4_rc(:,1),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C4_col = repmat(IND_C4_rc(:,2),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C4_chan = repmat([1:array_sz(indLayer-1,3)/2],length(IND_C4_rc(:,1)),1);
        tmp_IND_C4_chan = tmp_IND_C4_chan(:);
        tmp_IND_C4 = sub2ind(array_sz(indLayer-1,:),tmp_IND_C4_row,tmp_IND_C4_col,tmp_IND_C4_chan);
    else
        tmp_IND_C4_row = repmat(IND_C4_rc(:,1),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C4_col = repmat(IND_C4_rc(:,2),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C4_chan = repmat([1:array_sz(indLayer-1,3)/2]+array_sz(indLayer-1,3)/2,length(IND_C4_rc(:,1)),1);
        tmp_IND_C4_chan = tmp_IND_C4_chan(:);
        tmp_IND_C4 = sub2ind(array_sz(indLayer-1,:),tmp_IND_C4_row,tmp_IND_C4_col,tmp_IND_C4_chan);
    end
    
    tmp_IND_C4_row_w = repmat(tmp_IND_rc(:,1),array_sz(indLayer-1,3)/2,1);
    tmp_IND_C4_col_w = repmat(tmp_IND_rc(:,2),array_sz(indLayer-1,3)/2,1);
    tmp_IND_C4_chan_w = repmat([1:array_sz(indLayer-1,3)/2],length(tmp_IND_rc(:,1)),1);
    tmp_IND_C4_chan_w  = tmp_IND_C4_chan_w (:);
    tmp_IND_C4_w = sub2ind(array_sz_conv5,tmp_IND_C4_row_w,tmp_IND_C4_col_w,tmp_IND_C4_chan_w);
    
    [~, idx] = ismember(cell_list_L4, tmp_IND_C4);
    idx(idx == 0) = [];
    
    if reverse && length(idx)~=0
        tmp = setdiff(1:1:1728, idx);
        tmp = tmp(randperm(length(idx)));
        idx = tmp;
    end
    
    for tmp_idx=1:length(idx)
        tmp_row = tmp_IND_C4_row_w(idx(tmp_idx));
        tmp_col = tmp_IND_C4_col_w(idx(tmp_idx));
        tmp_chan = tmp_IND_C4_chan_w(idx(tmp_idx));
        W_conv5_ablation(tmp_row, tmp_col, tmp_chan, IND_C5_chan) = 0;
    end
end

net_ablation = net_rand;
net_tmp = net_ablation.saveobj;
net_tmp.Layers(14).Weights = W_conv5_ablation;
net_ablation = net_ablation.loadobj(net_tmp);
end