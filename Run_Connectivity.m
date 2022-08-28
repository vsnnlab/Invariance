%% Result 4
%  Computational model explains spontaneous emergence of invariance in untrained networks

%% Plot and analyize setting
var_array =  {'translation','scaling','rotation','viewpoint'};
vtype = find(not(cellfun('isempty',strfind(var_array, var_type))));

StrTitle = {'Translation','Scaling','Rotation','Viewpoint'};
StrUnit = {'r_R_F','%','deg', 'deg'};
arrayXlim = [-0.3, 75,-50,-60; 0.3,125,50,60];
StrXlabel = {'Translation (r_R_F)','Scaling (%)','Rotation (deg)', 'Viewpoint (deg)'};
StrYlabel = {'Effective range (r_R_F)','Effective range (%)','Effective range (deg)', 'Effective Range (deg)'};
var_axis = linspace(arrayXlim(1, vtype), arrayXlim(2, vtype), 13);


CONNECTIVITY_RESULT = cell(6, 2);

for nn=1:20
    % index : inv / spe / spe_pre
    ll = 5;
    layers_ind = [2,6,10,12,14];

    net_rand = Cell_Net{nn};
    W_conv5 = net_rand.Layers(layers_ind(ll)).Weights;

    conv4_invariant_unit_idx = Cell_Inv{ll-1, 1,nn};
    conv4_specific_unit_idx = Cell_Inv{ll-1, 2,nn};
    conv4_specific_unit_pref = Cell_Inv{ll-1, 3,nn};

    %% Conv5 Invariant Unit

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

    % From object-units / from non-object-units

    CONNECTIVITY_RESULT{1,1}(nn,:) = nanmean([inv_connectivity_mat{1, 1} inv_connectivity_mat{1, 2} inv_connectivity_mat{1, 3} ...
        inv_connectivity_mat{1, 4} inv_connectivity_mat{1, 5} inv_connectivity_mat{1, 6}]); % from object units
    CONNECTIVITY_RESULT{1,2}(nn,:) = nanmean(inv_connectivity_mat{1, 7}); % from non-object units
    CONNECTIVITY_RESULT{1,3}(nn,:) = [nanmean(inv_connectivity_mat{1, 2}) ...
        nanmean(inv_connectivity_mat{1, 3}) nanmean(inv_connectivity_mat{1, 4}) ...
        nanmean(inv_connectivity_mat{1, 5}) nanmean(inv_connectivity_mat{1, 6})]; % from specific units
       
    %% Conv5 Specific Unit
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

        CONNECTIVITY_RESULT{1+view,1}(nn,:) = nanmean([spec_connectivity_mat{1, 1} spec_connectivity_mat{1, 2} spec_connectivity_mat{1, 3} ...
            spec_connectivity_mat{1, 4} spec_connectivity_mat{1, 5} spec_connectivity_mat{1, 6}]);
        CONNECTIVITY_RESULT{1+view,2}(nn,:) = nanmean(spec_connectivity_mat{1, 7});
        CONNECTIVITY_RESULT{1+view,3}(nn,:) = [nanmean(spec_connectivity_mat{1, 2}) nanmean(spec_connectivity_mat{1, 3}) nanmean(spec_connectivity_mat{1, 4}) ...
            nanmean(spec_connectivity_mat{1, 5}) nanmean(spec_connectivity_mat{1, 6})];
    end
end

figure('units','normalized','outerposition',[0 0.5 1 0.5]); drawnow
sgtitle('Connectivity analysis between conv4 and conv5')

%% Invariant Unit
% From object-units / from non-object-units
ylim_min = -0.00;
ylim_max = 0.008;
ylim([ylim_min ylim_max]);

subplot(3, 3, 1);

hold on;
bar([nanmean(CONNECTIVITY_RESULT{1,1}) nanmean(CONNECTIVITY_RESULT{1,2})]);
errorbar([nanmean(CONNECTIVITY_RESULT{1,1}) nanmean(CONNECTIVITY_RESULT{1,2})], ...
    [nanstd(CONNECTIVITY_RESULT{1,1})./sqrt(NN) nanstd(CONNECTIVITY_RESULT{1,2})./sqrt(NN)]);
hold off;
xticks([1 2]);
xticklabels(["From object-units" "From non-object-units"]);
ylabel("Average of weights")
title("Projections from the source layer")
ylim([ylim_min ylim_max]);

subplot(3, 3, 2);
errorbar(nanmean(CONNECTIVITY_RESULT{1,3}), nanstd(CONNECTIVITY_RESULT{1,3})./sqrt(NN));
xticks([1 2 3 4 5]);
xticklabels(Cell_var_axis{vtype});
xlabel(strcat(StrTitle{vtype}, "-specific units (Conv4)"));
ylabel("Average of weights")
title("Projections from the source layer")
ylim([ylim_min ylim_max]);

%% Specific Unit
% From object-units / from non-object-units
subplot(3, 3, 4);

hold on;
bar([nanmean(CONNECTIVITY_RESULT{2,1}) nanmean(CONNECTIVITY_RESULT{2,2})]);
errorbar([mean(CONNECTIVITY_RESULT{2,1}) nanmean(CONNECTIVITY_RESULT{2,2})], ...
    [nanstd(CONNECTIVITY_RESULT{2,1})./sqrt(NN) nanstd(CONNECTIVITY_RESULT{2,2})./sqrt(NN)]);
hold off;
xticks([1 2]);
xticklabels(["From object-units" "From non-object-units"]);
ylabel("Average of weights")
title("Projections from the source layer")
ylim([ylim_min ylim_max]);

subplot(3, 3, 5);
hold on;
for view=1:5
    errorbar(nanmean(CONNECTIVITY_RESULT{1+view,3}), nanstd(CONNECTIVITY_RESULT{1+view,3})./sqrt(NN));
end
hold off;
xticks([1 2 3 4 5]);
xticklabels(Cell_var_axis{vtype});
xlabel(strcat(StrTitle{vtype}, "-specific units (Conv4)"));
ylabel("Average of weights")
title("Projections from the source layer");
ylim([ylim_min ylim_max]);
legend('-30 spec', '-15 spec', '0 spec', '+15 spec', '+30 spec');


subplot(3, 3, [3 6]);
heatmap_image = zeros(5, 5);
for view=1:5
    heatmap_image(view,:)=nanmean(CONNECTIVITY_RESULT{1+view,3});
end
%heatmap_image = heatmap_image ./ max(max(abs(heatmap_image)));
imagesc(heatmap_image);
xlabel('Conv4 Specific Unit');
xticks(1:5);
xticklabels({'-30 spec', '-15 spec', '0 spec', '+15 spec', '+30 spec'});
ylabel('Conv5 Specific Unit');
yticks(1:5);
yticklabels({'-30 spec', '-15 spec', '0 spec', '+15 spec', '+30 spec'});
load('Colorbar_Tsao.mat');
colormap(cmap);
%caxis([mean(heatmap_image(:))-1.2*std(heatmap_image(:)) mean(heatmap_image(:))+1.2*std(heatmap_image(:))]);
%caxis([0.002 0.0035]);
caxis([0.002 0.004]);
colorbar;

subplot(3, 3, 9);
specific_image = cell(9, 1);
for ii=1:5
    for jj=1:5
        specific_image{ii-jj+5} = [specific_image{ii-jj+5} heatmap_image(ii,jj)];
    end
end
s1 = plot(linspace(-60,60,9), ...
[mean(specific_image{1,1}) mean(specific_image{2,1}) mean(specific_image{3,1}) mean(specific_image{4,1}) ...
mean(specific_image{5,1}) mean(specific_image{6,1}) mean(specific_image{7,1}) mean(specific_image{8,1}) ...
mean(specific_image{9,1})],'k');
shadedErrorBar(linspace(-60,60,9), ...
[mean(specific_image{1,1}) mean(specific_image{2,1}) mean(specific_image{3,1}) mean(specific_image{4,1}) ...
mean(specific_image{5,1}) mean(specific_image{6,1}) mean(specific_image{7,1}) mean(specific_image{8,1}) ...
mean(specific_image{9,1})], ...
[std(specific_image{1,1}) std(specific_image{2,1}) std(specific_image{3,1}) std(specific_image{4,1}) ...
std(specific_image{5,1}) std(specific_image{6,1}) std(specific_image{7,1}) std(specific_image{8,1}) ...
std(specific_image{9,1})]./sqrt(NN),'lineprops',{'k','markerfacecolor','k'});
% line([arrayXlim(1,vtype)*2,arrayXlim(2,vtype)*2],[1 1],'Color','k','LineStyle','--')
xlim([arrayXlim(1,vtype)*2,arrayXlim(2,vtype)*2]); 
xticks(linspace(-60,60,9));
%ylim([0.5 1.5]);
xlabel('Distance of preferred angle from center'); ylabel('Weight');
%title('Untrained network');
ylim([ylim_min ylim_max]);

subplot(3, 3, 8);
hom_idx_inv = 1./nanstd(CONNECTIVITY_RESULT{1,3}, [], 2);
hom_idx_spe = mean([1./nanstd(CONNECTIVITY_RESULT{2,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{3,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{4,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{5,3}, [], 2), ...
    1./nanstd(CONNECTIVITY_RESULT{6,3}, [], 2)],2);
hom_idx_inv = hom_idx_inv./hom_idx_spe;
hom_idx_spe = hom_idx_spe./hom_idx_spe;

hold on;
errorbar([nanmean(hom_idx_inv),nanmean(hom_idx_spe)], ...
    [std(hom_idx_inv)/sqrt(NN), std(hom_idx_spe)/sqrt(NN)]);
hold off;
xticks([1 2]);
xticklabels(["Invariant units" "Specific units"]);
ylabel("Homogeneous index")


filename = strcat('RESULT_Connectivity_', target_class, '_',var_type, '.mat');
save(filename,'CONNECTIVITY_RESULT');
