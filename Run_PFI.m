%% Result 3
%  Viewpoint-invariant unit and specific units and its visual feature encoding

tic
load(['RAW_UnitInfo_',target_class,'_',var_type,'.mat'])
toc
 
%% Setting parameters
 
% Simulation parameter
NN = 20;                                                                   % number of networks for analysis
inpSize = 227;                                                             % width or height of each image in the object stimulus
 
% Network
layersSet = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};                 % names of feature extraction layers
array_sz = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256];         % dimensions of activation maps of each layer
layerArray = [1:5];                                                        % target layers
 
% Analysis                                                                           
pThr = 0.001;                                                              % p-value threshold of selective response
idxClass = 1;                                                              % index of face class in the dataset
 
%% Preferred feature images of face-selective units in untrained networks (Fig.2, Fig.S4) 
target_IND_Cell = cell(4, NN);
for nn=1:NN
    target_IND_Cell{1,nn} = Cell_Inv{5,1,nn};
    target_IND_Cell{2,nn} = Cell_Inv{5,2,nn}(Cell_Inv{5,3,nn}==1);
    target_IND_Cell{3,nn} = Cell_Inv{5,2,nn}(Cell_Inv{5,3,nn}==3);
    target_IND_Cell{4,nn} = Cell_Inv{5,2,nn}(Cell_Inv{5,3,nn}==5);
end
 
for target =1:4
    
   %% Reverse correlation (Fig 2a)
    % Stimulus parameters
    N_image = 2500;         % Number of stimulus images
    iteration = 100;         % Number of iteration
    img_size = 227;         % Image size
    dot_size = 5;           % Size of 2D Gaussian filter
 
    % Generate 2D Gaussian filters
    [pos_xx,pos_yy] = meshgrid(linspace(1+dot_size,img_size-dot_size,sqrt(N_image)),linspace(1+dot_size,img_size-dot_size,sqrt(N_image)));
    pos_xy_list = pos_xx(:) + 1i*pos_yy(:);
    [xx_field,yy_field] = meshgrid(1:img_size,1:img_size); xy_field = xx_field + 1i*yy_field;
 
    img_list = zeros(img_size,img_size,3,length(pos_xy_list));
    count = 1;
    for pp = 1:length(pos_xy_list)
        pos_tmp = pos_xy_list(pp);
        img_tmp = repmat(exp(-(abs(xy_field-pos_tmp).^2)/2/dot_size.^2)*0.3,1,1,3);
        img_list(:,:,:,count) = -img_tmp;
        count = count + 1;
    end
    Gau_stimulus = cat(4,img_list,-img_list);
    size(Gau_stimulus)
 
    % Iterative PFI calculation
    PFI = zeros(img_size,img_size,3)+255/2;                                         %Initial PFI
    PFI_mat = zeros(img_size,img_size,iteration+1); PFI_mat(:,:,1) = PFI(:,:,1);
 
    figure; sgtitle([target_class,'_',num2str(target)]);
    set(gcf,'Position', get(0,'Screensize'));
    
    for iter = 1:iteration
        tic
        disp(num2str(iter));
        PFI_0 = PFI; % Save previous PFI
 
        % Generate stimulus as a summation of previous PFI and gaussian stimulus
        IMG = repmat(PFI/255,[1,1,1,size(Gau_stimulus,4)])+Gau_stimulus;
        IMG = uint8(IMG*255);     IMG(IMG<0) = 0; IMG(IMG>255) = 255;
        
        act_collection = [];
        for nn=1:NN
            net_rand = Cell_Net{nn};
            % Measure the response of random AlexNet
            act_rand = activations(net_rand,IMG,'relu5');       % Response of 'relu5' layer in random AlexNet
            act_reshape = reshape(act_rand,43264,size(IMG,4));  % Reshape the response in 2D form
            act_reshape_sel = act_reshape(target_IND_Cell{target,nn},:);          % Find the response of face-selective neurons
            act_collection = [act_collection; act_reshape_sel];
        end
        mean_act = mean(act_collection,1);                 % Average response of face-selective neurons
 
        % Calculate the PFI
        norm_act_reshape = repmat(permute(mean_act-min(mean_act),[1,3,4,2]),img_size,img_size,3);
        PFI = sum(norm_act_reshape.*double(IMG),4)/sum(mean_act-min(mean_act));
        PFI_diff = PFI-PFI_0; PFI = PFI_0 + PFI_diff*10;
        PFI(PFI<0) = 0; PFI(PFI>255) = 255;
        PFI_mat(:,:,iter+1) = PFI(:,:,1);
        
        subplot(ceil(sqrt(iteration)),ceil(sqrt(iteration)),iter)
        imagesc(PFI_mat(:,:,iter+1));axis image off; colormap gray; caxis([0 255]); drawnow
        toc
    end
    tmp_PFI = PFI_mat(:,:,end);
    save(strcat("PFI_",num2str(target),'_220825.mat'),"tmp_PFI");
end

