
function [Cell_IND_conv4,Cell_W_conv4] = fun_BackTrack_L5toL4(cell_list_L5,array_sz,net_rand,W_conv5)
indLayer = 5;
array_sz_conv5 = size(W_conv5);
[c,r] = meshgrid(1:size(W_conv5,1),1:size(W_conv5,2));
IND_rc = [r(:), c(:)];

Cell_IND_conv4 = cell(length(cell_list_L5),1);
Cell_W_conv4 = cell(length(cell_list_L5),1);

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
    
    Cell_W_conv4{ii} = t_w(tmp_IND_C4_w);
    Cell_IND_conv4{ii} = tmp_IND_C4;
end

end