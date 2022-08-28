function [STIM,temp_STR_LABEL] = fun_GetStim(stim_type, var_type, target_class, subset)

%% Set base directory
if strcmp(var_type, 'None')
    base_directory = ['Image/' stim_type '/'];
else
    base_directory = ['Image/' stim_type '/' var_type '/'];
end

STR_LABEL = {'bed','chair','desk','dresser','monitor', ...
        'night_stand', 'sofa', 'table', 'toilet'};                         % label of classes in the object stimulus
    
%% Selectivity Stimulus
if strcmp(stim_type, 'selectivity') || strcmp(stim_type, 'selectivity_var')

    temp_STR_LABEL = STR_LABEL;
    temp_STR_LABEL(ismember(temp_STR_LABEL, target_class)) = [];
    temp_STR_LABEL = [target_class, temp_STR_LABEL, strcat(target_class, '_scrambled')];

    for jj=1:length(temp_STR_LABEL)
        path_directory = [base_directory char(temp_STR_LABEL(jj)) '/'];
        files = dir([path_directory '*.png']);
        for ii=1:length(files)
            current_filename = files(ii).name;
            current_image = imread([path_directory current_filename]);
            STIM(:,:,:,(jj-1)*length(files)+ii) = repmat(current_image, 1, 1, 3); % Duplicate single channal image to 3 channal image
        end
    end

%% SVM Stimulus
elseif strcmp(stim_type, 'SVM') || strcmp(stim_type, 'SVM_var')
    base_directory = [base_directory '/' subset '/'];
    numIMGtot = 120;
    %STIM = zeros(227,227,3,numIMGtot);
    
    % First half: Target Class Set
    path_directory = [base_directory char(target_class) '/'];
    files = dir([path_directory '*.png']);
    for ii=1:numIMGtot/2
        current_filename = files(ii).name;
        current_image = imread([path_directory current_filename]);
        STIM(:,:,:,ii) = repmat(current_image, 1, 1, 3); % Duplicate single channal image to 3 channal image
    end
    
    % Last half: Non-Target Class Set
    temp_STR_LABEL = STR_LABEL;
    temp_STR_LABEL(ismember(temp_STR_LABEL, target_class)) = [];
    for jj=1:length(temp_STR_LABEL)
        path_directory = [base_directory char(temp_STR_LABEL(jj)) '/'];
        files = dir([path_directory '*.png']);
        img_to_load = ceil(numIMGtot/2/(length(temp_STR_LABEL)+1));
        
        for ii=1:img_to_load
            current_filename = files(ii).name;
            current_image = imread([path_directory current_filename]);
            STIM(:,:,:,numIMGtot/2+(jj-1)*img_to_load+ii) = repmat(current_image, 1, 1, 3); % Duplicate single channal image to 3 channal image
        end
    end
    
    path_directory = [base_directory char(strcat(target_class, '_scrambled')) '/'];
    files = dir([path_directory '*.png']);
    for ii=1:numIMGtot/2-ceil(numIMGtot/2/(length(temp_STR_LABEL)+1))*length(temp_STR_LABEL)
        current_filename = files(ii).name;
        current_image = imread([path_directory current_filename]);
        STIM(:,:,:,numIMGtot/2+length(temp_STR_LABEL)*img_to_load+ii) = ...
            repmat(current_image, 1, 1, 3);                                % Duplicate single channal image to 3 channal image
    end
    
    temp_STR_LABEL = {target_class char(strcat('non_', target_class))};
    
%% Invariance Test
elseif strcmp(stim_type, 'invariance_test')
    % invariance_test stimulus set label
    if strcmp(var_type, 'rotation')
        VAR_idx = {'-180.0', '-150.0', '-120.0', '-90.0', '-60.0', '-30.0', ...
            '0.0', '30.0' '60.0' '90.0', '120.0', '150.0', '180.0'};
    elseif strcmp(var_type, 'scaling')
        VAR_idx = {'0.25', '0.375', '0.5', '0.625', '0.75', '0.875', ...
            '1.0', '1.125', '1.25', '1.375', '1.5', '1.625', '1.75'};
    elseif strcmp(var_type, 'translation')
        VAR_idx = {'-0.9', '-0.75', '-0.6', '-0.45', '-0.3', '-0.15', ...
            '0.0', '0.15' '0.3' '0.45', '0.6', '0.75', '0.9'};
    elseif strcmp(var_type, 'viewpoint')
        VAR_idx = {'-90.0', '-75.0', '-60.0', '-45.0', '-30.0', '-15.0', ...
            '0.0', '15.0', '30.0', '45.0', '60.0', '75.0', '90.0'};
    end
    
    temp_STR_LABEL = STR_LABEL;
    temp_STR_LABEL(ismember(temp_STR_LABEL, target_class)) = [];
    temp_STR_LABEL = [target_class, temp_STR_LABEL];
    
    for kk=1:length(temp_STR_LABEL)
        for jj=1:length(VAR_idx)
            path_directory = [base_directory char(temp_STR_LABEL(kk)) '/' char(VAR_idx(jj)) '/'];
            files = dir([path_directory '*.png']);
            for ii=1:length(files)
                current_filename = files(ii).name;
                current_image = imread([path_directory current_filename]);
                file_idx = length(VAR_idx)*length(files)*(kk-1)+length(files)*(jj-1)+ii;
                STIM(:,:,file_idx) = current_image;
                var_cls_idx(file_idx, 1)=jj;
                var_cls_idx(file_idx, 2)=kk;
            end
        end
    end
    
    temp_STR_LABEL = var_cls_idx;
    
%% Invariance Unit
elseif strcmp(stim_type, 'invariance_unit')
    % invariance_unit stimulus set label
    if strcmp(var_type, 'rotation')
        VAR_idx = {'-50.0', '-25.0', '0.0', '25.0', '50.0'};
    elseif strcmp(var_type, 'scaling')
        VAR_idx = {'0.75', '0.875', '1.0', '1.125', '1.25'};
    elseif strcmp(var_type, 'translation')
        VAR_idx = {'-0.3', '-0.15', '0.0', '0.15', '0.3'};
    elseif strcmp(var_type, 'viewpoint')
        VAR_idx = {'-60.0', '-30.0', '0.0', '30.0', '60.0'};
    end
    
    for jj=1:length(VAR_idx)
        path_directory = [base_directory char(target_class) '/' char(VAR_idx(jj)) '/'];
        files = dir([path_directory '*.png']);
        for ii=1:length(files)
            current_filename = files(ii).name;
            current_image = imread([path_directory current_filename]);
            STIM(:,:,:,length(files)*(jj-1)+ii) = repmat(current_image, 1, 1, 3); % Duplicate single channal image to 3 channal image
        end
    end
    
    temp_STR_LABEL = VAR_idx;
end

end



