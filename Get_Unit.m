% Randomly initialized AlexNet and object-selective units

%% Setting parameters

% Simulation parameter
inpSize = 227;                                                             % width or height of each image in the object stimulus

% Network
layersSet = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};	               % names of feature extraction layers
array_sz = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256];         % dimensions of activation maps of each layer
layerArray = [1:5];                                                        % target layers
 
% Analysis                                                                           
pThr = 0.001;                                                              % p-value threshold of selective response

%% Loading pretrained Alexnet and image dataset
disp(['Load imageset and networks ...'])

tic
net = alexnet;                                                             % pretained AlexNet                       
if strcmp(var_type, 'None')
    [IMG_ORI, STR_LABEL] = fun_GetStim('selectivity', var_type, target_class); 
else
    [IMG_ORI, STR_LABEL] = fun_GetStim('selectivity_var', var_type, target_class); 
end
IMG_ORI = single(IMG_ORI);
numCLS = length(STR_LABEL);
numIMG = 200;
idxClass = 1;
toc

%% Check unit index file already saved
filename = strcat('./Result/RAW_UnitInfo_', target_class, '_', var_type, '.mat');

if exist(filename) == 0
%% Measure response and find object-selective unit
disp(['Find object unit in untrained network ...'])
tic
Cell_Net = cell(NN,1);
Cell_Idx = cell(NN,length(layerArray));
for nn = 1:NN
    net_rand = fun_Initializeweight(net,1,1);
    for ll = 3:length(layerArray)
        % Measuring responses of neurons in the target layer
        num_cell = prod(array_sz(layerArray(ll),:));
        act_rand = activations(net_rand,IMG_ORI,layersSet{layerArray(ll)});

        % Finding selective neuron to target class
        [cell_idx] = fun_FindNeuron(act_rand,num_cell,numCLS,numIMG,pThr,idxClass);
        Cell_Idx{nn,ll} = cell_idx; clearvars cell_idx
    end
    Cell_Net{nn} = net_rand; clearvars act_rand net_rand
end
toc

%% Single unit level invariance characteristic
disp(['Find effective range and preferred angle ...']);

% Load feature variant stimulus set
[IMG_var, var_cls_idx] = fun_GetStim('invariance_test', var_type, target_class);
IMG_var = single(repmat(permute(IMG_var,[1 2 4 3]),[1 1 3]));
var_idx = var_cls_idx(:, 1); cls_idx = var_cls_idx(:, 2); clearvars var_cls_idx;

Cell_Info = cell(NN,2);
tic
for nn=1:NN
    %% Analysis for invariance to image variation (Fig.S5a-f)
    net_rand = Cell_Net{nn};                                               % untrained AlexNet  
    Idx_Target = Cell_Idx{nn,length(layerArray)};                          % indices of face units in the untrained AlexNet
    num_cell = prod(array_sz(layerArray(length(layerArray)),:));
 
    %% Measure network response 
    act_rand = activations(net_rand,IMG_var,layersSet{layerArray(length(layerArray))});
    act_re = reshape(act_rand,num_cell,size(IMG_var,4));
    act_face = act_re(Idx_Target,:);
    num_face_cell = size(Idx_Target,1);
    clearvars act_rand act_re
 
    %% Measure effective range
    [resp_z_mat,face_resp_z_mat,max_resp_z_mat] = fun_InvRange_Resp(act_face,num_face_cell,cls_idx,var_idx);
    
    threshold = 0.05;
    effective_range = zeros(size(face_resp_z_mat, 1), 1);
    max_angle = zeros(size(face_resp_z_mat, 1), 1);

    for ii=1:size(face_resp_z_mat, 1) 
        % find preferred angle
        max_angle(ii) = find(mean(face_resp_z_mat(ii,:,:),3)==max(mean(face_resp_z_mat(ii,:,:), 3)));
        % find effective range
        temp_p_value = zeros(13, 1);
        for jj=1:13
            try
                temp_p_value(jj) = ranksum(squeeze(face_resp_z_mat(ii,jj,:)),...
                    squeeze(max_resp_z_mat(ii,jj,:)), 'tail', 'right');
            catch
                temp_p_value(jj) = 1;
            end
        end

        max_range = 0;
        for jj=1:13
            if temp_p_value(jj) < threshold
                for kk=jj+1:13
                    if temp_p_value(kk) >= threshold
                        break
                    end
                end
                if kk-jj+1 > max_range
                    max_range = kk-jj+1;
                end
            end
        end
        effective_range(ii) = max_range;
    end
    
    Cell_Info{nn,1} = effective_range;
    Cell_Info{nn,2} = max_angle; 
end
toc

%% Find viewpoint specfic / invariant object units
disp(['Find viewpoint specific and invariant untis ...']);
Cell_Inv = cell(length(layerArray),4, nn); % index : inv / spe / spe_pre
IMG_viewpoint = fun_GetStim('invariance_unit', var_type, target_class);    % viewpoint stimulus
IMG_view = single(IMG_viewpoint); clearvars IMG_viewpoint   
numIMG_view = 50;                                                          % number of images of a class in the object stimulus
numCLS_view = 5;                                                           % number of classes in the viewpoint stimulus          
tic
for nn=1:NN
    net_rand = Cell_Net{nn};                                               % untrained AlexNet  
    arrayClass = [ones(numIMG_view,1);2.*ones(numIMG_view,1);3.*ones(numIMG_view,1);4.*ones(numIMG_view,1);5.*ones(numIMG_view,1)];
 
    Cell_view_p = cell(5,1);
    Cell_view_pref = cell(5,1);
    Cell_view_peak = cell(5,1);
    Cell_view_stat = cell(5,1);
 
    for ll = 3:5
        indLayer  = ll;
        num_cell = prod(array_sz(indLayer,:));
        Idx_Target = Cell_Idx{nn,ll};
 
        act_rand = activations(net_rand,IMG_view,layersSet{indLayer});
        act_reshape = reshape(act_rand,num_cell,size(IMG_view,4)); clearvars act_rand 
        act_reshape3D = zeros(num_cell,numCLS_view,numIMG_view);
        for cc = 1:numCLS_view
            act_reshape3D(:,cc,:) = act_reshape(:,(cc-1)*numIMG_view+1:cc*numIMG_view);
        end
        act_reshape = act_reshape(Idx_Target,:);
        act_reshape3D = act_reshape3D(Idx_Target,:,:); clearvars act_rand 
 
        p = zeros(length(Idx_Target),1); % p-value
        pref = zeros(length(Idx_Target),numCLS_view); % preferred angle
        peak = zeros(length(Idx_Target),numCLS_view); % peak location
        vstat = zeros(length(Idx_Target),1);
 
        for ii = 1:length(Idx_Target)
            p(ii) = anova1(act_reshape(ii,:),arrayClass,'off');
 
            meanFR = squeeze(mean(act_reshape3D(ii,:,:),3))';
            
            firstPeak = max(meanFR);
            peakView = find(meanFR==max(meanFR));
            secondPeak = meanFR(meanFR~=max(meanFR));
            
            peakView = find(meanFR==max(meanFR));
            peak(ii,peakView) = 1;

            if sum(meanFR) == 0
                pref(ii,:) = 0;
            else
                [~,pref(ii,:)] = sort(meanFR,'descend');
                vstat(ii,1) = std(meanFR);
            end
        end
 
        Cell_view_p{ll} = p;
        Cell_view_pref{ll} = pref;
        Cell_view_peak{ll} = peak;
        Cell_view_stat{ll} = vstat;
    end
 
    for ll = 3:5
        idx = Cell_Idx{nn,ll};
        p = Cell_view_p{ll};
        pref = Cell_view_pref{ll};
        peak = Cell_view_peak{ll};
 
        Cell_Inv{ll,1,nn} = transpose(idx((pref(:,1) ~= 0)&(p(:,1)>=0.05))); % invariant unit index
        Cell_Inv{ll,2,nn} = transpose(idx((pref(:,1) ~= 0)&(p(:,1)<0.05)&(sum(peak,2) == 1))); % specific unit index
        Cell_Inv{ll,3,nn} = transpose(pref((pref(:,1) ~= 0)&(p(:,1)<0.05)&(sum(peak,2) == 1))); % specific unit preferred angle
        Cell_Inv{ll,4,nn} = Cell_view_stat{ll}; % specific unit stat
    end
end

save(filename,'Cell_Net', 'Cell_Idx','Cell_Info', 'Cell_Inv', 'NN');       % save files

else
disp(['Load saved object unit in untrained network ...'])
load(filename)
end

toc