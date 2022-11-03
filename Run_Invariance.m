% Result 2a: Invariance observed in untrained networks

%% Plot and analyize setting
var_array =  {'translation','scaling','rotation','viewpoint'};
vtype = find(not(cellfun('isempty',strfind(var_array, var_type))));

StrTitle = {'Translation','Scaling','Rotation','Viewpoint'};
StrUnit = {'r_R_F','%','deg', 'deg'};
arrayXlim = [-1.5, 25,-180,-90; 1.5,175,180,90];
StrXlabel = {'Translation (r_R_F)','Scaling (%)','Rotation (deg)', 'Viewpoint (deg)'};
StrYlabel = {'Effective range (r_R_F)','Effective range (%)','Effective range (deg)', 'Effective Range (deg)'};
var_axis = linspace(arrayXlim(1, vtype), arrayXlim(2, vtype), 13);

nn=1;                                                                      % single network to analyize

%% Load feature variant stimulus set and plot example

[IMG_var, var_cls_idx] = fun_GetStim('invariance_test', var_type, target_class);
IMG_var = single(repmat(permute(IMG_var,[1 2 4 3]),[1 1 3]));
var_idx = var_cls_idx(:, 1); cls_idx = var_cls_idx(:, 2); clearvars var_cls_idx;

figure('units','normalized','outerposition',[0 0 1 1]); drawnow
sgtitle(['Invariance observed in untrained networks'])

%% Viewpoint variant stimulus (Figure 2B, left)
show_cls = [1 6 8];
show_var = 1:3:13;
for jj = 1:3
    for ii = 1:5
        temp_idx = find((var_idx == show_var(ii)) & (cls_idx == show_cls(jj)) == 1);
        subplot(7, 13, ii + 13 * (jj-1));
        imagesc(IMG_var(:,:,1,temp_idx(1)));
        colormap('gray');
        xticks([]); yticks([]);
        if ii==1; ylabel(char(STR_LABEL(show_cls(jj)))); end
        if jj==3; xlabel(var_axis(show_var(ii))); end
    end
end

%% Measure invariant response for invariance to image variation
net_rand = Cell_Net{nn};                                                   % untrained AlexNet  
Idx_Target = Cell_Idx{nn,length(layerArray)};                              % indices of object units in the untrained AlexNet
num_cell = prod(array_sz(layerArray(length(layerArray)),:));

act_rand = activations(net_rand,IMG_var,layersSet{layerArray(length(layerArray))});
act_re = reshape(act_rand,num_cell,size(IMG_var,4));
act_target = act_re(Idx_Target,:);
num_target_cell = size(Idx_Target,1);
clearvars act_rand

single_target = Cell_Idx{nn,5}(and((Cell_Info{nn,1}==7),(Cell_Info{nn,2}==7)));
act_single = act_re(single_target(1),:);

%% Single unit response to viewpoint-variant stimulus (Figure 2C, left)
[single_resp_z_mat,single_target_resp_z_mat,single_max_resp_z_mat] = fun_InvRange_Resp(act_single,1,cls_idx,var_idx);
single_non_target = squeeze(single_resp_z_mat(:,7,:,:));

x1 = squeeze(single_target_resp_z_mat(:,7,:));
x2 = squeeze(single_target_resp_z_mat(:,10,:));
x3 = squeeze(single_target_resp_z_mat(:,13,:));
x4 = single_non_target(:);

subplot(7, 13, [7 8 9 20 21 22 33 34 35]);
hold on
    errorbar(1:4, [mean(x1), mean(x2), mean(x3), mean(x4)], ...
        [std(x1)/sqrt(200), std(x2)/sqrt(200), std(x3)/sqrt(200), std(x4)/sqrt(200)], '-', color='k');
    plot(1:4, [mean(x1), mean(x2), mean(x3), mean(x4)], 'o', color='r');
hold off
yline(0, '--');
xticks([1:4]); xlim([0.5 4.5]);
xticklabels({'deg=0', 'deg=45', 'deg=90', 'non-target'});
ylabel('Response (z-scored)')
set(gca,'TickLabelInterpreter','none','TickDir','out');
ylim([-0.8 1.2]);
title('Single unit response to viewpoint-variant stimulus');

%% Single unit tuning curve for viewpoint-variant stimulus (Figure 2C, right)
single_target_resp_z_mat(single_target_resp_z_mat==inf) = nan;
single_target_resp_z_mean = squeeze(nanmean(single_target_resp_z_mat,1));
single_max_resp_z_mean = squeeze(nanmean(single_max_resp_z_mat,1));

subplot(7, 13, [11 12 13 24 25 26 37 38 39]);
hold on;
shadedErrorBar(var_axis,nanmean(single_target_resp_z_mean,2),nanstd(single_target_resp_z_mean,0, 2)./sqrt(200),'lineprops','r');
shadedErrorBar(var_axis,nanmean(single_max_resp_z_mean,2),nanstd(single_max_resp_z_mean,0,2)./sqrt(200),'lineprops','k');
s1 = plot(var_axis,nanmean(single_target_resp_z_mean,2),'color',[1 0 0]);
s2 = plot(var_axis,nanmean(single_max_resp_z_mean,2),'color',[0 0 0]);
hold off;
xlim([arrayXlim(1,vtype),arrayXlim(2,vtype)]); 
ylim([-0.8 1.2]);
xlabel(StrXlabel{vtype});ylabel('Response (z-scored)')
title('Single unit tuning curve for viewpoint-variant stimulus');
legend([s1,s2],{'Target Object stimulus','Non-target Object stimulus'},'Location','northeast')

%% Measure effective range
[resp_z_mat,target_resp_z_mat,max_resp_z_mat] = fun_InvRange_Resp(act_target,num_target_cell,cls_idx,var_idx);

% Response of face units to image variation
target_resp_z_mat(target_resp_z_mat==inf) = nan;
target_resp_z_mean = squeeze(nanmean(target_resp_z_mat,1));
max_resp_z_mean = squeeze(nanmean(max_resp_z_mat,1));

% Find effective range
threshold = 0.05;
effective_range = zeros(size(target_resp_z_mean, 1), 1);
for ii=1:size(target_resp_z_mean, 1) 
    try; effective_range(ii) = ranksum(squeeze(target_resp_z_mean(ii,:)),squeeze(max_resp_z_mean(ii,:)), 'tail', 'right');
    catch; effective_range(ii) = 1;
    end
end
er_unit_resp = [min(find(effective_range<threshold)) - 0.5 max(find(effective_range<threshold)) + 0.5];

%% Effective range of raw image correlation (Figure 2D, left)
subplot(7, 13, [53 54 55 66 67 68 79 80 81]);
hold on;
shadedErrorBar(var_axis,nanmean(target_resp_z_mean,2),nanstd(target_resp_z_mean,0, 2),'lineprops','r');
shadedErrorBar(var_axis,nanmean(max_resp_z_mean,2),nanstd(max_resp_z_mean,0,2),'lineprops','k');
s1 = plot(var_axis,nanmean(target_resp_z_mean,2),'color',[1 0 0]);
s2 = plot(var_axis,nanmean(max_resp_z_mean,2),'color',[0 0 0]);
hold off;
xlim([arrayXlim(1,vtype),arrayXlim(2,vtype)]); 
ylim([-0.8 1.2]);
xlabel(StrXlabel{vtype});ylabel('Response (z-scored)')
title('Unit responses');
xline((er_unit_resp-7)*15, '--');
legend([s1,s2],{'Target Object stimulus','Non-target Object stimulus'},'Location','northeast')

clearvars IMG_cls IMG_var

%% Calculate raw image correlation
filename = strcat('./Result/RESULT_Invariance_RawImageCorrelation_', target_class, '_', var_type, '.mat');

if exist(filename) == 0
[corr_mat,corr_cont_mat] = fun_InvRange_ImgCorr(IMG_var,cls_idx);
save(filename, 'corr_mat','corr_cont_mat');

else
load(filename);
end

% find effective range
effective_range = zeros(size(corr_mat, 2), 1);
for ii=1:size(corr_mat, 2) 
    try; effective_range(ii) = ranksum(squeeze(corr_mat(:,ii)),squeeze(corr_cont_mat(:,ii)), 'tail', 'right');
    catch; effective_range(ii) = 1;
    end
end
er_img_corr = [min(find(effective_range<threshold)) - 0.5 max(find(effective_range<threshold)) + 0.5];

%% Effective range of raw image correlation (Figure 2D, right)
subplot(7, 13, [57 58 59 71 72 73 83 84 85]);
hold on;
shadedErrorBar(var_axis,mean(corr_mat,1),std(corr_mat,1),'lineprops','r');
shadedErrorBar(var_axis,mean(corr_cont_mat,1),std(corr_cont_mat,1),'lineprops','k');
s1 = plot(var_axis,mean(corr_mat,1),'color',[1 0 0]);
s2 = plot(var_axis,mean(corr_cont_mat,1),'color',[0 0 0]);
xlim([arrayXlim(1,vtype),arrayXlim(2,vtype)]); 
hold off;
ylim([-0.8 1.2]);
xlabel(StrXlabel{vtype});ylabel('Correlation coefficient')
title('Raw image correlation');
xline((er_img_corr-7)*15, '--');
legend([s1,s2],{'Target Object stimulus','Non-target Object stimulus'},'Location','northeast')

%% Effective range of invariance (Figure 2F)
subplot(7, 13, [62 63 64 65 75 76 77 78 88 89 90 91]);
bar(1:2, [er_unit_resp(2)-er_unit_resp(1) er_img_corr(2)-er_img_corr(1)] .* 15);
ylim([0 180]); ylabel('Effective range (deg)');
xticks([1:2]); xticklabels({'Unit\newline{response}', 'Raw image\newline{correlation}'});
title('Effective range of invariance');