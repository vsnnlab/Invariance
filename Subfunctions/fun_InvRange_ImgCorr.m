function [corr_mat,corr_cont_mat] = fun_InvRange_ImgCorr(IMG_var,cls_idx)

class_idx = cls_idx == 1;
IMG_cls = squeeze(IMG_var(:,:,1,class_idx));
corr_mat = zeros(200,13);
corr_cont_mat = zeros(200,13);

for ii = 1:200
    IMG_cen = IMG_cls(:,:,(ii-1)*13+7);
    
    for vv = 1:13
        IMG_temp = IMG_cls(:,:,(ii-1)*13+vv);
        corr_temp = corrcoef(IMG_cen,IMG_temp);
        corr_mat(ii,vv) = corr_temp(1,2);
        
        corr_cont_temp = zeros(8,1);
        for kk=2:9
            IMG_cont_group_temp = squeeze(IMG_var(:,:,1,cls_idx ==kk));
            IMG_temp = IMG_cont_group_temp(:,:,(ii-1)*13+vv);
            tmp = corrcoef(IMG_cen,IMG_temp);
            corr_cont_temp(kk-1,1) = tmp(1,2);
        end
        corr_cont_mat(ii,vv) = max(corr_cont_temp);
    end
end

end

