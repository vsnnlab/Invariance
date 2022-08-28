%% Result 2
%  Invariance observed in untrained networks

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

%% Load feature variant stimulus set

[IMG_var, var_cls_idx] = fun_GetStim('invariance_test', var_type, target_class);
IMG_var = single(repmat(permute(IMG_var,[1 2 4 3]),[1 1 3]));
var_idx = var_cls_idx(:, 1); cls_idx = var_cls_idx(:, 2); clearvars var_cls_idx;

%% Measure invariant response for invariance to image variation
net_rand = Cell_Net{nn};                                                   % untrained AlexNet  
Idx_Target = Cell_Idx{nn,length(layerArray)};                              % indices of object units in the untrained AlexNet
num_cell = prod(array_sz(layerArray(length(layerArray)),:));

act_rand = activations(net_rand,IMG_var,layersSet{layerArray(length(layerArray))});
act_re = reshape(act_rand,num_cell,size(IMG_var,4));
act_target = act_re(Idx_Target,:);
num_target_cell = size(Idx_Target,1);
clearvars act_rand

%% Plot figure for each image variation
figure('units','normalized','outerposition',[0 0 1 1]); drawnow
sgtitle(['Invariant charateristics of object-selective units (',StrTitle{vtype},')'])

single_target = Cell_Idx{nn,5}(and((Cell_Info{nn,1}==7),(Cell_Info{nn,2}==7)));
act_single = act_re(single_target(1),:);

% box plot
[single_resp_z_mat,single_target_resp_z_mat,single_max_resp_z_mat] = fun_InvRange_Resp(act_single,1,cls_idx,var_idx);
single_non_target = squeeze(single_resp_z_mat(:,7,:,:));

x1 = squeeze(single_target_resp_z_mat(:,7,:));
x2 = squeeze(single_target_resp_z_mat(:,10,:));
x3 = squeeze(single_target_resp_z_mat(:,13,:));
x4 = single_non_target(:);

subplot(1, 3, 1);
errorbar(([mean(x1), mean(x2), mean(x3), mean(x4)]), [std(x1)/sqrt(200), std(x2)/sqrt(200), std(x3)/sqrt(200), std(x4)/sqrt(200)]);
yline(0, '--');
xticklabels({'deg=0', 'deg=45', 'deg=90', 'non-target'});
set(gca,'TickLabelInterpreter','none','TickDir','out');
ylim([-0.8 1.2]);

% Response of face units to image variation
single_target_resp_z_mat(single_target_resp_z_mat==inf) = nan;
single_target_resp_z_mean = squeeze(nanmean(single_target_resp_z_mat,1));
single_max_resp_z_mean = squeeze(nanmean(single_max_resp_z_mat,1));

subplot(1, 3, 2);
hold on;
shadedErrorBar(var_axis,nanmean(single_target_resp_z_mean,2),nanstd(single_target_resp_z_mean,0, 2),'lineprops','r');
shadedErrorBar(var_axis,nanmean(single_max_resp_z_mean,2),nanstd(single_max_resp_z_mean,0,2),'lineprops','k');
s1 = plot(var_axis,nanmean(single_target_resp_z_mean,2),'color',[1 0 0]);
s2 = plot(var_axis,nanmean(single_max_resp_z_mean,2),'color',[0 0 0]);
hold off;
xlim([arrayXlim(1,vtype),arrayXlim(2,vtype)]); 
ylim([-2 2]);
xlabel(StrXlabel{vtype});ylabel('Response (z-scored)')
title('Response of object-selective units'); legend([s1,s2],{'Target Object stimulus','Non-target Object stimulus'},'Location','northeast')

%% Average response

%% Measure effective range
[resp_z_mat,target_resp_z_mat,max_resp_z_mat] = fun_InvRange_Resp(act_target,num_target_cell,cls_idx,var_idx);

% Response of face units to image variation
target_resp_z_mat(target_resp_z_mat==inf) = nan;
target_resp_z_mean = squeeze(nanmean(target_resp_z_mat,1));
max_resp_z_mean = squeeze(nanmean(max_resp_z_mat,1));

subplot(1, 3, 3);
hold on;
shadedErrorBar(var_axis,nanmean(target_resp_z_mean,2),nanstd(target_resp_z_mean,0, 2),'lineprops','r');
shadedErrorBar(var_axis,nanmean(max_resp_z_mean,2),nanstd(max_resp_z_mean,0,2),'lineprops','k');
s1 = plot(var_axis,nanmean(target_resp_z_mean,2),'color',[1 0 0]);
s2 = plot(var_axis,nanmean(max_resp_z_mean,2),'color',[0 0 0]);
hold off;
xlim([arrayXlim(1,vtype),arrayXlim(2,vtype)]); 
ylim([-1 1]);
xlabel(StrXlabel{vtype});ylabel('Response (z-scored)')
title('Response of object-selective units'); legend([s1,s2],{'Target Object stimulus','Non-target Object stimulus'},'Location','northeast')

threshold = 0.05;
effective_range = zeros(size(target_resp_z_mean, 1), 1);

for ii=1:size(target_resp_z_mean, 1) 
    % find effective range
    try
        effective_range(ii) = ranksum(squeeze(target_resp_z_mean(ii,:)),squeeze(max_resp_z_mean(ii,:)), 'tail', 'right');
    catch
        effective_range(ii) = 1;
    end
end
range_start = min(find(effective_range<threshold)) - 0.5;
range_end = max(find(effective_range<threshold)) + 0.5;
xline(([range_start range_end]-7)*15);

clearvars IMG_cls IMG_var
