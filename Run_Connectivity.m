% Result 3: Computational model explains spontaneous emergence of invariance in untrained networks

%% Plot and analyize setting
var_array =  {'translation','scaling','rotation','viewpoint'};
vtype = find(not(cellfun('isempty',strfind(var_array, var_type))));

StrTitle = {'Translation','Scaling','Rotation','Viewpoint'};
StrUnit = {'r_R_F','%','deg', 'deg'};
arrayXlim = [-0.3, 75,-50,-60; 0.3,125,50,60];
StrXlabel = {'Translation (r_R_F)','Scaling (%)','Rotation (deg)', 'Viewpoint (deg)'};
StrYlabel = {'Effective range (r_R_F)','Effective range (%)','Effective range (deg)', 'Effective Range (deg)'};
var_axis = linspace(arrayXlim(1, vtype), arrayXlim(2, vtype), 5);


%% Perform connectivity analysis or load saved result
filename = strcat('./Result/RESULT_Connectivity_', target_class, '_', var_type, '.mat');

if exist(filename) == 0
disp(['Perform connectivity analysis ...'])

CONNECTIVITY_RESULT = cell(6, 2);
ll = 5;
layers_ind = [2,6,10,12,14];

for nn=1:NN
    % index : inv / spe / spe_pre
    net_rand = Cell_Net{nn};
    W_conv5 = net_rand.Layers(layers_ind(ll)).Weights;

    conv4_invariant_unit_idx = Cell_Inv{ll-1, 1,nn};
    conv4_specific_unit_idx = Cell_Inv{ll-1, 2,nn};
    conv4_specific_unit_pref = Cell_Inv{ll-1, 3,nn};

    % Conv5 Invariant Unit
    conv5_invariant_unit_idx = Cell_Inv{ll, 1,nn};
    [Cell_IND_conv4,Cell_W_conv4] = fun_BackTrack_L5toL4(conv5_invariant_unit_idx,array_sz,net_rand,W_conv5);

    inv_connectivity_mat = cell(1,7); % index: inv / spe(5) / non-obj unit

    for ii=1:length(conv5_invariant_unit_idx)
        for jj=1:length(Cell_IND_conv4{ii})
            if ismember(Cell_IND_conv4{ii}(jj), conv4_invariant_unit_idx)
                inv_connectivity_mat{1, 1} = [inv_connectivity_mat{1, 1} Cell_W_conv4{ii}(jj)];
            elseif ismember(Cell_IND_conv4{ii}(jj), conv4_specific_unit_idx)
                temp_pref = conv4_specific_unit_pref(conv4_specific_unit_idx==Cell_IND_conv4{ii}(jj));
                inv_connectivity_mat{1, 1+temp_pref} = [inv_connectivity_mat{1, 1+temp_pref} Cell_W_conv4{ii}(jj)];
            else
                inv_connectivity_mat{1, 7} = [inv_connectivity_mat{1, 7} Cell_W_conv4{ii}(jj)];
            end
        end
    end

    % From object-units / from non-object-unit
    CONNECTIVITY_RESULT{1,1}(nn,:) = nanmean([inv_connectivity_mat{1, 1} inv_connectivity_mat{1, 2} inv_connectivity_mat{1, 3} ...
        inv_connectivity_mat{1, 4} inv_connectivity_mat{1, 5} inv_connectivity_mat{1, 6}]); % from object units
    CONNECTIVITY_RESULT{1,2}(nn,:) = nanmean(inv_connectivity_mat{1, 7}); % from non-object units
    CONNECTIVITY_RESULT{1,3}(nn,:) = [nanmean(inv_connectivity_mat{1, 2}) ...
        nanmean(inv_connectivity_mat{1, 3}) nanmean(inv_connectivity_mat{1, 4}) ...
        nanmean(inv_connectivity_mat{1, 5}) nanmean(inv_connectivity_mat{1, 6})]; % from specific units
       
    % Conv5 Specific Unit
    for view=1:5
        conv5_specific_unit_idx = Cell_Inv{ll, 2,nn};
        conv5_specific_unit_idx = conv5_specific_unit_idx(find(Cell_Inv{ll, 3,nn} == view));
        [Cell_IND_conv4,Cell_W_conv4] = fun_BackTrack_L5toL4(conv5_specific_unit_idx,array_sz,net_rand,W_conv5);

        spec_connectivity_mat = cell(1,7); % index: inv / spe(5) / non-obj unit

        for ii=1:length(conv5_specific_unit_idx)
            for jj=1:length(Cell_IND_conv4{ii})
                if ismember(Cell_IND_conv4{ii}(jj), conv4_invariant_unit_idx)
                    spec_connectivity_mat{1, 1} = [spec_connectivity_mat{1, 1} Cell_W_conv4{ii}(jj)];
                elseif ismember(Cell_IND_conv4{ii}(jj), conv4_specific_unit_idx)
                    temp_pref = conv4_specific_unit_pref(conv4_specific_unit_idx==Cell_IND_conv4{ii}(jj));
                    spec_connectivity_mat{1, 1+temp_pref} = [spec_connectivity_mat{1, 1+temp_pref} Cell_W_conv4{ii}(jj)];
                else
                    spec_connectivity_mat{1, 7} = [spec_connectivity_mat{1, 7} Cell_W_conv4{ii}(jj)];
                end
            end
        end

        % From object-units / from non-object-units

        CONNECTIVITY_RESULT{1+view,1}(nn,:) = nanmean([spec_connectivity_mat{1, 1} ...
            spec_connectivity_mat{1, 2} spec_connectivity_mat{1, 3} spec_connectivity_mat{1, 4} ...
            spec_connectivity_mat{1, 5} spec_connectivity_mat{1, 6}]);
        CONNECTIVITY_RESULT{1+view,2}(nn,:) = nanmean(spec_connectivity_mat{1, 7});
        CONNECTIVITY_RESULT{1+view,3}(nn,:) = [nanmean(spec_connectivity_mat{1, 2}) ...
            nanmean(spec_connectivity_mat{1, 3}) nanmean(spec_connectivity_mat{1, 4}) ...
            nanmean(spec_connectivity_mat{1, 5}) nanmean(spec_connectivity_mat{1, 6})];
    end
end
save(filename,'CONNECTIVITY_RESULT');

else
disp(['Load saved result of connectivity analysis ...'])
load(filename)
end

figure('units','normalized','outerposition',[0 0.7 1 0.7]); drawnow
sgtitle('Emergence of viewpoint invariance based on unbiased projection from viewpoint-specific units')

ylim_min = -0.0005;
ylim_max = 0.006;

% Projection to viewpoint specific unit (Conv5)
%% From object-units and non-object-units (Conv4) to specific units (Conv5) (Figure 4A, middle)
subplot(2, 4, 1);
errorbar([mean(CONNECTIVITY_RESULT{2,1}) nanmean(CONNECTIVITY_RESULT{2,2})], ...
    [nanstd(CONNECTIVITY_RESULT{2,1})./sqrt(NN) nanstd(CONNECTIVITY_RESULT{2,2})./sqrt(NN)], ...
    'o', color=color_gray,MarkerFaceColor=color_gray);
xticks([1 2]);
xticklabels(["From object-units" "From non-object-units"]);
ylabel("Average of weights")
title("Projections to specific units (Conv5)")
ylim([ylim_min ylim_max]); xlim([0.5 2.5]); yline(0, '--');

%% From specific units (Conv4) to specific units (Conv5) (Figure 4A, right)
subplot(2, 4, [2 3]);
hold on;
for view=3
    errorbar(nanmean(CONNECTIVITY_RESULT{1+view,3}), nanstd(CONNECTIVITY_RESULT{1+view,3})./sqrt(NN), ...
        '-o', color=color_blue,MarkerFaceColor=color_blue);
end
hold off;
xticks([1 2 3 4 5]);
xticklabels(var_axis);
xlabel(strcat(StrTitle{vtype}, "-specific units (Conv4)"));
ylabel("Average of weights")
title("Projections to specific units (Conv5)");
ylim([ylim_min ylim_max]); xlim([0.5 5.5]); yline(0, '--');

%% Connectivity between specific units (Figure 4B)
subplot(2, 4, 4);
heatmap_image = zeros(5, 5);
for view=1:5
    heatmap_image(6-view,:)=nanmean(CONNECTIVITY_RESULT{1+view,3});
end

imagesc(heatmap_image);
xlabel('Specific Units (Conv4)');
xticks(1:5);
xticklabels(var_axis);
ylabel('Specific Units (Conv5)');
yticks(1:5);
yticklabels(flip(var_axis));
load('Colorbar_Tsao.mat');
colormap(cmap);
caxis([mean(heatmap_image(:))-1.2*std(heatmap_image(:)) mean(heatmap_image(:))+1.2*std(heatmap_image(:))]);
colorbar;
title('Connectivity between specific units');

% Projection to viewpoint specific unit (Conv5)
%% From object-units and non-object-units (Conv4) to invariant units (Conv5) (Figure 4C, middle)
subplot(2, 4, 5);
errorbar([nanmean(CONNECTIVITY_RESULT{1,1}) nanmean(CONNECTIVITY_RESULT{1,2})], ...
    [nanstd(CONNECTIVITY_RESULT{1,1})./sqrt(NN) nanstd(CONNECTIVITY_RESULT{1,2})./sqrt(NN)], ...
    'o', color=color_gray,MarkerFaceColor=color_gray);
xticks([1 2]);
xticklabels(["From object-units" "From non-object-units"]);
ylabel("Average of weights")
title("Projections to invariant units (Conv5)")
ylim([ylim_min ylim_max]); xlim([0.5 2.5]); yline(0, '--');

%% From specific units (Conv4) to invariant units (Conv5) (Figure 4C, right)
subplot(2, 4, [6 7]);
errorbar(nanmean(CONNECTIVITY_RESULT{1,3}), nanstd(CONNECTIVITY_RESULT{1,3})./sqrt(NN), ...
    '-o', color=color_red,MarkerFaceColor=color_red);
xticks([1 2 3 4 5]);
xticklabels(var_axis);
xlabel(strcat(StrTitle{vtype}, "-specific units (Conv4)"));
ylabel("Average of weights")
title("Projections to invariant units (Conv5)")
ylim([ylim_min ylim_max]); xlim([0.5 5.5]); yline(0, '--');

%% Homogeneity in projection weights (Figure 4D)
subplot(2, 4, 8);
hom_idx_inv = 1./nanstd(CONNECTIVITY_RESULT{1,3}, [], 2);
hom_idx_spe = mean([1./nanstd(CONNECTIVITY_RESULT{2,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{3,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{4,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{5,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{6,3}, [], 2)],2);
hom_idx_inv = hom_idx_inv./hom_idx_spe;
hom_idx_spe = hom_idx_spe./hom_idx_spe;

errorbar([nanmean(hom_idx_inv),nanmean(hom_idx_spe)], ...
    [std(hom_idx_inv)/sqrt(NN), std(hom_idx_spe)/sqrt(NN)], ...
    '-o', color=color_gray,MarkerFaceColor=color_gray);
xticks([1 2]);
xticklabels(["Invariant units" "Specific units"]);
ylabel("Homogeneous index")
title("Homogeneity in projection weights");
xlim([0.5 2.5]); ylim([0.5 3]);