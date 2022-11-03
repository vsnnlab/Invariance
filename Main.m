% Demo code for <Invariance of object detection in untrained deep neural networks>

%% Information
%==================================================================================================================================================
% Invariance of object detection in untrained deep neural networks
% Jeonghwan Cheon, Seungdae Baek, and Se-Bum Paik*
% https://doi.org/10.3389/fncom.2022.1030707
%
% *Contact: sbpaik@kaist.ac.kr
%
% Demo code ver. November 03. 2022
%
% Prerequirement 
% 1) MATLAB 2021b or later version
% 2) Installation of the Deep Learning Toolbox
%    (https://www.mathworks.com/products/deep-learning.html)
% 3) Installation of the pretrained AlexNet
%    (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)
% 4) Please download 'Image.zip' from below link
%    (https://doi.org/10.5281/zenodo.7276304)
%    and unzip these files in the same directory
% 5) Decide whether to use pre-stored results
%      - If you want to simulate all of results, please delete 'Result' directory
%        (Raw image correlation, obtained PFI, SVM result will be provided)
%      - Else, all of results will be generate anew.
%        (Raw image correlation > 4 hours, PFI result > 1 days, SVM result > 1 hour)

% Output of the code
% Below results for untrained AlexNet will be shown.
% Result 1) Emergence of selectivity to various objects in untrained networks
% Result 2) Invariance observed in untrained networks
% Result 3) Viewpoint-invariant unit and specific units and its visual feature encoding
% Result 4) Computational model explains spontaneous emergence of invariance in untrained networks
% Result 5) Invariantly tuned unit responses enable invariant object detection
%==================================================================================================================================================

close all;clc;clear;
seed = 1; rng(seed)                                                        % fixed random seed for regenerating same result

addpath('Image')
addpath('Subfunctions')
toolbox_chk;                                                               % checking matlab version and toolbox
color_set;                                                                 % load preset color

%% Setting analysis
% select analysis type
var_type = 'viewpoint';                                                    % invariance type to analysis (only 'viewpoint' available)
target_class = 'toilet';                                                   % object (only 'toilet' available)

% Demo code
res1 = 0; res2a = 1; res2b = 0; res3 = 0; res4 = 0;                        % flag for analysis corresponding each figure 
NN = 3;                                                                    % number of networks for analysis (20 networks used in manuscript)

%% Finding selective neuron to target class
Get_Unit;

%% Result 1) Emergence of selectivity to various objects in untrained networks
if res1 == 1
disp('Result 1 ...')
tic
Run_Selectivity;
toc
end

%% Result 2a) Invariance observed in untrained networks
if res2a == 1
disp('Result 2a ...')
tic
Run_Invariance;
toc
end

%% Result 2b) Viewpoint-invariant unit and specific units and its visual feature encoding
if res2b == 1
disp('Result 2b ...')
tic
Run_PFI;
toc
end

%% Result 3) Computational model explains spontaneous emergence of invariance in untrained networks
if res3 == 1
disp('Result 3 ...')
tic
Run_Connectivity;
toc
end

%% Result 4) Invariantly tuned unit responses enable invariant object detection
if res4 == 1
disp('Result 4 ...')
tic
Run_SVM;
toc
end