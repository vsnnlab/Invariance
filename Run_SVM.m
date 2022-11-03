% Result 4: Invariantly tuned unit responses enable invariant object detection

%% Perform SVM analysis or load saved result
filename = strcat('./Result/RESULT_SVM_', target_class, '_', var_type, '.mat');

if exist(filename) == 0
disp(['Perform PFI analysis ...'])

RESULT_SVM = cell(5,1);

for nn=1:NN
    tic
    %% Train SVM using response of an untrained network
    net_rand = Cell_Net{nn};                                                % untrained AlexNet                                                    
    num_cell = prod(array_sz(layerArray(length(layerArray)),:));
    
    reN = 50;

    %% Find neuron
    disp('Prepare SVM training for object detection task ... (~ 40 sec)')
    act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
    actORIre = reshape(act_rand,num_cell,size(IMG_ORI,4));

    Idx_All = [1:num_cell]';
    Idx_Target = Cell_Idx{nn,length(layerArray)};
    Idx_Inv = Cell_Inv{5,1,nn};
    Idx_Spe0 = Cell_Inv{5,2,nn}(Cell_Inv{5,3,nn} == 3);

    numNeuron = min([length(Idx_Target), length(Idx_Inv), length(Idx_Spe0)]);

    p = zeros(num_cell,1);
    for mm = 1:num_cell
        meanRF =[mean(actORIre(mm,1:numIMG)),mean(actORIre(mm,numIMG+1:2*numIMG)),mean(actORIre(mm,2*numIMG+1:3*numIMG)),...
            mean(actORIre(mm,3*numIMG+1:4*numIMG)),mean(actORIre(mm,4*numIMG+1:5*numIMG)),mean(actORIre(mm,5*numIMG+1:6*numIMG))...
            ,mean(actORIre(mm,6*numIMG+1:7*numIMG)),mean(actORIre(mm,7*numIMG+1:8*numIMG)),mean(actORIre(mm,8*numIMG+1:9*numIMG))...
            ,mean(actORIre(mm,9*numIMG+1:10*numIMG))];
        if sum(meanRF) == 0
            p(mm) = 1;
        else
            [~,cls_max] = max(meanRF); [~,cls_min] = min(meanRF);
            p(mm) = ranksum(actORIre(mm,numIMG*(cls_max-1)+1:numIMG*cls_max),actORIre(mm,numIMG*(cls_min-1)+1:numIMG*cls_min));
        end
    end
    arrayClass = [ones(numIMG,1);2.*ones(numIMG,1);3.*ones(numIMG,1);4.*ones(numIMG,1);5.*ones(numIMG,1);...
        6.*ones(numIMG,1);7.*ones(numIMG,1);8.*ones(numIMG,1);9.*ones(numIMG,1);10.*ones(numIMG,1);];
    p2 = zeros(num_cell,1); for mm = 1:num_cell;p2(mm) = anova1(actORIre(mm,:),arrayClass,'off');end
    Idx_NS = intersect(find(p>0.9),find(p>0.9));
    
    %% Task1: Test1 vs Test 2 (Specific unit and invariant unit)
    % Invariant, Test 1
    cell_list = Idx_Inv;
    array_SVM = zeros(1,reN);
    for ii = 1:reN
        array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
            length(layerArray),numNeuron, var_type, target_class, 120, 120);
    end
    RESULT_SVM{1}(nn, 1) = mean(array_SVM);
    
    % Invariant, Test 2
    cell_list = Idx_Inv;
    array_SVM = zeros(1,reN);
    for ii = 1:reN
        array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
            length(layerArray), numNeuron, var_type, target_class, 0, 120);
    end
    RESULT_SVM{1}(nn, 2) = mean(array_SVM);
    
    % Specific, Test 1
    cell_list = Idx_Spe0;
    array_SVM = zeros(1,reN);
    for ii = 1:reN
        array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
            length(layerArray), numNeuron, var_type, target_class, 120, 120);
    end
    RESULT_SVM{1}(nn, 3) = mean(array_SVM);
    
    % Specific, Test 2
    cell_list = Idx_Spe0;
    array_SVM = zeros(1,reN);
    for ii = 1:reN
        array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
            length(layerArray), numNeuron, var_type, target_class, 0, 120);
    end
    RESULT_SVM{1}(nn, 4) = mean(array_SVM);
    
    % Conv5 
    cell_list = Idx_All;
    array_SVM = zeros(1,reN);
    for ii = 1:reN
        array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
            length(layerArray),length(cell_list), var_type, target_class, 0, 120);
    end
    RESULT_SVM{1}(nn, 5) = mean(array_SVM);
    
    % NS
    cell_list = Idx_NS;
    Cell_SVM_mult_center = cell(3,1);                                  
    array_SVM = zeros(1,reN);
    for ii = 1:reN
        array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
            length(layerArray), numNeuron, var_type, target_class, 0, 120);
    end
    RESULT_SVM{1}(nn, 6) = mean(array_SVM);
    
    view_array = 0:30:180;
    for view_idx=1:length(view_array)
        Conv5 
        
        cell_list = Idx_All;
        array_SVM = zeros(1,reN);
        for ii = 1:reN
            array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
                length(layerArray),length(cell_list), var_type, target_class, 0, view_array(view_idx));
        end
        RESULT_SVM{2}(nn, view_idx) = mean(array_SVM);
        
        % Invariant, Test 2
        cell_list = Idx_Inv;
        array_SVM = zeros(1,reN);
        for ii = 1:reN
            array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
                length(layerArray), numNeuron, var_type, target_class, 0, view_array(view_idx));
        end
        RESULT_SVM{2}(nn, view_idx) = mean(array_SVM);
    
        % All spe
        cell_list = Cell_Inv{5,2,nn};
        array_SVM = zeros(1,reN);
        for ii = 1:reN
            array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
                length(layerArray), numNeuron, var_type, target_class, 0, view_array(view_idx));
        end
        RESULT_SVM{3}(nn, view_idx) = mean(array_SVM);
        
        % Specific, Test 2
        cell_list = Idx_Spe0;
        array_SVM = zeros(1,reN);
        for ii = 1:reN
            array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
                length(layerArray), numNeuron, var_type, target_class, 0, view_array(view_idx));
        end
        RESULT_SVM{4}(nn, view_idx) = mean(array_SVM);
        
        % NS
        cell_list = Idx_NS;
        array_SVM = zeros(1,reN);
        for ii = 1:reN
            array_SVM(1,ii) = fun_SVM_Var(net_rand,num_cell,cell_list,layersSet,...
                length(layerArray), numNeuron, var_type, target_class, 0, view_array(view_idx));
        end
        RESULT_SVM{5}(nn, view_idx) = mean(array_SVM);
    end
    toc
end
save(filename, 'RESULT_SVM');
else
load(filename);
end

figure('units','normalized','outerposition',[0 0.7 0.7 0.7]); drawnow
sgtitle('Invariantly tuned unit responses enable invariant object detection')

data_length = size(RESULT_SVM{1},1);

%% SVM trainte with multi- or single-viewpoint stimuli (Figure 5B)
subplot(2, 2, 1);

face_color = {'white', color_red, 'white', color_blue};
edge_color = {color_red, 'white', color_blue, 'white'};
hold on;
for ii=1:4
    data = mean(RESULT_SVM{1}(:,ii));
    err = std(RESULT_SVM{1}(:,ii));
    h = bar(ii, data);
    set(h,'FaceColor',char(face_color(ii)), 'EdgeColor', char(edge_color(ii)), 'LineWidth', 1.5);
    
    er = errorbar(ii, data, err, err);
    er.Color = [0 0 0];
    er.LineStyle = 'none';
end
hold off;
ylim([0.4 1.0]);
yline(0.5, '--', 'Chance level');
ylabel('Correct ratio');
xticks(1:4);
xticklabels({'Train1', 'Train2', 'Train1', 'Train2'});
title('SVM trainte with multi- or single-viewpoint stimuli');

%% SVM trained with single-viewpoint stimuli (Train 2, Figure 5C)
subplot(2, 2, 2);
face_color = {color_red, color_blue, color_gray};
plot_target = [2 4 6];
hold on
for ii=1:3
    data = mean(RESULT_SVM{1}(:,plot_target(ii)));
    err = std(RESULT_SVM{1}(:,plot_target(ii)));
    h = bar(ii, data);
    set(h,'FaceColor',char(face_color(ii)));
    
    er = errorbar(ii, data, err, err);
    er.Color = [0 0 0];
    er.LineStyle = 'none';
end
hold off;
ylim([0.4 1.0]);
yline(0.5, '--', 'Chance level');
ylabel('Correct ratio');
yline(mean(RESULT_SVM{1}(:,5)), '--', 'All units (Conv5)')
xticks(1:3);
xticklabels({'Invariant\newline{units}', 'Specific\newline{units (0°)}', 'Non-selective\newline{units}'});
title('SVM trained with single-viewpoint stimuli');

%% SVM trained with single-viewpoint stimuli test by various ranges (Train 2, Figure 5D)
subplot(2, 2, 3);
summary = [];
face_color = {'r', 'b', 'b', 'k'};
hold on;
for jj=1:4
    shadedErrorBar(0:30:180, mean(RESULT_SVM{1+jj}(:,:)), std(RESULT_SVM{1+jj}(:,:)),'Lineprops',char(face_color(jj)));
    summary = [summary, RESULT_SVM{1+jj}(:,7)];
end
hold off;
ylim([0.4 1.0]);
xticks(0:30:180);
xlabel('Test image variation range (deg)');
ylabel('Correct ratio');
title('SVM trained with single-viewpoint stimuli test by various ranges');
legend({'','Invariant\newline{units}','','All specific\newline{units}', ...
    '','Specific\newline{units (0°)}', '','Non-selective\newline{units}'}, ...
    'Location', 'southwest');

%% Performance using different types of unit (Train 2, Figure 5E)
subplot(2, 2, 4);
hold on;
    errorbar(1:4, mean(summary), std(summary), '-', color='k');
    plot(1:4, mean(summary), 'o', color='r');
hold off;
xlim([0.5 4.5]);
ylim([0.4 1.0]);
xticks(1:4);
xticklabels({'Invariant\newline{units}','All specific\newline{units}', ...
    'Specific\newline{units (0°)}', 'Non-selective\newline{units}'});
ylabel('Correct ratio');
title('Performance using different types of unit');