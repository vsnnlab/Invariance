function [Cell_IND_conv3,Cell_W_conv3] = fun_BackTrack_L4toL3(cell_list_L4,array_sz,net_rand,W_conv4)
indLayer = 4;
array_sz_conv4 = size(W_conv4);
[c,r] = meshgrid(1:size(W_conv4,1),1:size(W_conv4,2)); IND_rc = [r(:), c(:)];

Cell_IND_conv3 = cell(length(cell_list_L4),1); Cell_W_conv3 = cell(length(cell_list_L4),1);
for ii = 1:length(cell_list_L4)
    tmp_IND_rc = IND_rc;
    ind_C4_cell = cell_list_L4(ii);
    [~,~,IND_C4_chan] = ind2sub(array_sz(indLayer,:),ind_C4_cell);
    t_w = W_conv4(:,:,:,IND_C4_chan); t_w = t_w(:);
    
    [IND_C3_rc] = fun_backtracking_C4toC3_XY(net_rand,ind_C4_cell);
    
    i1 = find((IND_C3_rc(:,1)<=0|IND_C3_rc(:,1)>array_sz(indLayer-1,1)));
    if length(i1)>0
        IND_C3_rc(i1,:) = []; tmp_IND_rc(i1,:) = [];
    end
    i2 = find((IND_C3_rc(:,2)<=0|IND_C3_rc(:,2)>array_sz(indLayer-1,1)));
    if length(i2)>0
        IND_C3_rc(i2,:) = []; tmp_IND_rc(i2,:) = [];
    end
    
    if IND_C4_chan <= array_sz(indLayer,3)/2
        tmp_IND_C3_row = repmat(IND_C3_rc(:,1),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C3_col = repmat(IND_C3_rc(:,2),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C3_chan = repmat([1:array_sz(indLayer-1,3)/2],length(IND_C3_rc(:,1)),1); tmp_IND_C3_chan = tmp_IND_C3_chan(:);
        tmp_IND_C3 = sub2ind(array_sz(indLayer-1,:),tmp_IND_C3_row,tmp_IND_C3_col,tmp_IND_C3_chan);
    else
        tmp_IND_C3_row = repmat(IND_C3_rc(:,1),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C3_col = repmat(IND_C3_rc(:,2),array_sz(indLayer-1,3)/2,1);
        tmp_IND_C3_chan = repmat([1:array_sz(indLayer-1,3)/2]+array_sz(indLayer-1,3)/2,length(IND_C3_rc(:,1)),1); tmp_IND_C3_chan = tmp_IND_C3_chan(:);
        tmp_IND_C3 = sub2ind(array_sz(indLayer-1,:),tmp_IND_C3_row,tmp_IND_C3_col,tmp_IND_C3_chan);
    end
    
    tmp_IND_C3_row_w = repmat(tmp_IND_rc(:,1),array_sz(indLayer-1,3)/2,1);
    tmp_IND_C3_col_w = repmat(tmp_IND_rc(:,2),array_sz(indLayer-1,3)/2,1);
    tmp_IND_C3_chan_w = repmat([1:array_sz(indLayer-1,3)/2],length(tmp_IND_rc(:,1)),1); tmp_IND_C3_chan_w  = tmp_IND_C3_chan_w (:);
    tmp_IND_C3_w = sub2ind(array_sz_conv4,tmp_IND_C3_row_w,tmp_IND_C3_col_w,tmp_IND_C3_chan_w);
    
    Cell_W_conv3{ii} = t_w(tmp_IND_C3_w);
    Cell_IND_conv3{ii} = tmp_IND_C3;
end

end