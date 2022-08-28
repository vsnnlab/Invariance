%% Result 5
%  Invariantly tuned unit responses enable invariant object detection

%% Detection of face images using the response of face units in untrained networks (Fig.3, Fig.S11-12) 
RESULT_SVM = cell(5,1);

for nn=1:NN
    tic
    %% Train SVM using response of an untrained network (Fig 3b-d)
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
    
    %%% Task1: Test1 vs Test 2 (Specific unit and invariant unit)
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
    
    view_array = [180];
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
    
    filename = strcat('RESULT_SVM_', target_class, '_',var_type, '.mat');
    save(filename,'RESULT_SVM');
end

%{
figure('units','normalized','outerposition',[0 0.5 1 0.5]); drawnow
sgtitle('Object detection task using the response of object-selective units')

subplot(1, 3, 1);
errorbar([1, 2, 3, 4], ...
    [mean(SVM_Result{1, 1}), mean(SVM_Result{1, 2}), mean(SVM_Result{1, 3}), mean(SVM_Result{1, 4})], ...
    [std(SVM_Result{1, 1}), std(SVM_Result{1, 2}), std(SVM_Result{1, 3}), std(SVM_Result{1, 4})])
xticks([1, 2, 3, 4]);
xticklabels({'Train1/Inv', 'Train2/Inv', 'Train1/Spec', 'Train2/Spec'});
yline(0.5, '--');
ylabel("Correct ratio");

subplot(1, 3, 2);
errorbar([1, 2, 3, 4], ...
    [mean(SVM_Result{2, 1}), mean(SVM_Result{1, 2}), mean(SVM_Result{1, 4}), mean(SVM_Result{2, 2})], ...
    [std(SVM_Result{2, 1}), std(SVM_Result{1, 2}), std(SVM_Result{1, 4}), std(SVM_Result{2, 2})])
xticks([1, 2, 3, 4]);
xticklabels({'All units', 'Inv', 'Spe', 'NS'});
yline(0.5, '--');
ylabel("Correct ratio");

subplot(1, 3, 3);
plot([0, 30], [mean(SVM_Result{2, 1}), mean(SVM_Result{1, 2}), mean(SVM_Result{1, 4}), mean(SVM_Result{2, 2})], ...
    [std(SVM_Result{2, 1}), std(SVM_Result{1, 2}), std(SVM_Result{1, 4}), std(SVM_Result{2, 2})])
xticks([1, 2, 3, 4]);
xticklabels(['All units', 'Inv', 'Spe', 'NS']);
yline(0.5, '--');
ylabel("Correct ratio");
%}

