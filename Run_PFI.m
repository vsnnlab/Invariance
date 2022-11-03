% Result 2b: Viewpoint-invariant unit and specific units and its visual feature encoding

%% Perform PFI analysis or load saved result
filename = strcat('./Result/RESULT_PFI_', target_class, '_', var_type, '.mat');

if exist(filename) == 0
disp(['Perform PFI analysis ...'])

PFI_RESULT = cell(4, 1);

% Set target units to analyze PFI
target_IND_Cell = cell(4, NN);
for nn=1:NN
    target_IND_Cell{1,nn} = Cell_Inv{5,1,nn};                              % invariant units
    target_IND_Cell{2,nn} = Cell_Inv{5,2,nn}(Cell_Inv{5,3,nn}==1);         % specific units (pref = -60)
    target_IND_Cell{3,nn} = Cell_Inv{5,2,nn}(Cell_Inv{5,3,nn}==3);         % specific units (pref = 0)
    target_IND_Cell{4,nn} = Cell_Inv{5,2,nn}(Cell_Inv{5,3,nn}==5);         % specific units (pref = +60)
end

% Reverse correlation
for target =1:4
    % Stimulus parameters
    N_image = 2500;                                                        % Number of stimulus images
    iteration = 100;                                                       % Number of iteration
    img_size = 227;                                                        % Image size
    dot_size = 5;                                                          % Size of 2D Gaussian filter
 
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
    PFI = zeros(img_size,img_size,3)+255/2;                                % Initial PFI
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
            act_rand = activations(net_rand,IMG,'relu5');                  % Response of 'relu5' layer in random AlexNet
            act_reshape = reshape(act_rand,43264,size(IMG,4));             % Reshape the response in 2D form
            act_reshape_sel = act_reshape(target_IND_Cell{target,nn},:);   % Find the response of face-selective neurons
            act_collection = [act_collection; act_reshape_sel];
        end
        mean_act = mean(act_collection,1);                                 % Average response of face-selective neurons
 
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

    PFI_RESULT{target} = PFI_mat(:,:,end);
end
save(filename, 'PFI_RESULT');

else
disp(['Load saved result of PFI analysis ...'])
load(filename);
end

%% Load feature variant stimulus set

[IMG_var, var_cls_idx] = fun_GetStim('invariance_test', var_type, target_class);
IMG_var = single(repmat(permute(IMG_var,[1 2 4 3]),[1 1 3]));
var_idx = var_cls_idx(:, 1); cls_idx = var_cls_idx(:, 2); clearvars var_cls_idx;

figure('units','normalized','outerposition',[0 0.5 1 0.5]); drawnow
sgtitle('Viewpoint-invariant unit and specific units and its visual feature encoding')

%% Visualize PFI and stimulus (Figure 3F)
view_arr = [3, 7, 10];

for ii = 1:4
    if ii==4; img_idx = find(cls_idx == 1);
    else; img_idx = find(and(var_idx == view_arr(ii), cls_idx == 1));
    end
    
    if ii ~= 4
        ax1 = subplot(3, 13, ii);
        imagesc(IMG_var(:,:,1,img_idx(1)));
        colormap(ax1, 'gray');
        xticks([]); yticks([]);
    end
        
    ax2 = subplot(3, 13, 13+ii);
    imagesc(mean(IMG_var(:,:,1,img_idx),4));
    colormap(ax2, 'gray');
    xticks([]); yticks([]);
    
    ax3 = subplot(3, 13, 26+ii);
    imagesc(PFI_RESULT{ii});
    colormap(ax3, 'gray');
    xticks([]); yticks([]);
    
    switch ii
        case 1
            ylabel(ax1, 'Example\newline{Stimulus}');
            ylabel(ax2, 'Averaged\newline{Stimulus}');
            ylabel(ax3, 'PFI');
            xlabel(ax3, '-60° spec.');
        case 2; xlabel(ax3,  '0° spec.');
        case 3; xlabel(ax3,  '60° spec.');
        case 4; xlabel(ax3,  'Invariant\newline{units}');
    end
end

%% Correlations between stimulus and PFI (Figure 3G)
stim_correlation = zeros(13, 4);

for kk = 1:4
    for ii = 1:13
        tmp_corr_mat = zeros(200, 1);
        tmp_idx = find(and(var_idx == ii, cls_idx == 1));
        for jj=1:200
            tmp_image = IMG_var(:,:,1,tmp_idx(jj));
            tmp = corrcoef(PFI_RESULT{kk},tmp_image);
            tmp_corr_mat(jj) = tmp(1,2);
        end
        stim_correlation(14-ii,kk) = mean(tmp_corr_mat);
    end
end

ax4 = subplot(3, 13, [6 7 8 9 19 20 21 22 32 33 34 35]);
imagesc(stim_correlation);
load('Colorbar_Tsao.mat');
colormap(ax4, cmap);
caxis([-0.02 0.06]);
colorbar;
xline(3.5, '--');
xticks(1:4); xticklabels({'-60° spec.' '0° spec.' '60° spec.' 'Invariant\newline{units}'});
xlabel('Preferred feature image');
yticks(1:2:13); yticklabels({flip(-90:30:90)});
ylabel('Stimulus viewpoint (deg)');
title('Correlations between stimulus and PFI');

%% Combined PFIs of specific units vs. PFIs of invariant units (Figure 3H)
x = 0:0.05:1;
y = 0:0.05:1;

[X,Y] = meshgrid(x,y);
Z=1-X-Y;
C = zeros(size(Z));
for ii = 1:length(x)
    for jj = 1:length(y)
        xx = x(ii);
        yy = y(jj);
        zz = 1-xx-yy;
            tmp_image = PFI_RESULT{1} * zz + PFI_RESULT{2} * yy + PFI_RESULT{3} * xx;
            tmp_image = (tmp_image - min(min(tmp_image)))./(max(max(tmp_image)) - min(min(tmp_image)));
            tmp = corrcoef(tmp_image,PFI_RESULT{4});
            C(ii,jj) = tmp(1,2);
    end
end

ax5 = subplot(3, 13, [11 12 13 24 25 26 37 38 39]);
surf(X, Y, Z, C);
zlim([0 1]);
c = colorbar; colormap(ax5, cmap)
xlabel('a'); ylabel('b'); zlabel('c');
title('Combined PFIs of specific units vs. PFIs of invariant units');