%% Demo code ver. 08/26/2022
%==================================================================================================================================================
% Invariance of object detection in untrained deep neural networks
% Jeonghwan Cheon, Seungdae Baek, and Se-Bum Paik*
%
% *Contact: sbpaik@kaist.ac.kr
%
% Prerequirement 
% 1) MATLAB 2021b or later version
% 2) Installation of the Deep Learning Toolbox
%    (https://www.mathworks.com/products/deep-learning.html)
% 3) Installation of the pretrained AlexNet
%    (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)
% 3) Please download 'Image.zip' from below link
%
%      - [Data URL] : https://drive.google.com/drive/folders/11KLOOW4QfqVQn0DqwGdXoUTyB0xxhmtw
%
%    and unzip these files in the same directory

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

%% Setting analysis
% select analysis type
var_type = 'viewpoint';                                                    % invariance type to analysis (only 'viewpoint' available)
target_class = 'toilet';                                                   % object 

% Demo code
res1 = 0; res2 = 1; res3 = 0; res4 = 0; res5 = 0;                          % flag for analysis corresponding each figure 
NN = 3;                                                                    % number of networks for analysis

%% Finding selective neuron to target class
Get_Unit;

%% Result 1) Emergence of selectivity to various objects in untrained networks
if res1 == 1
disp('Result 1 ...')
tic
Run_Selectivity;
toc
end

%% Result 2) Invariance observed in untrained networks
if res2 == 1
disp('Result 2 ...')
tic
Run_Invariance;
toc
end

%% Result 3) Viewpoint-invariant unit and specific units and its visual feature encoding
if res3 == 1
disp('Result 3 ...')
tic
Run_PFI;
toc
end

%% Result 4) Computational model explains spontaneous emergence of invariance in untrained networks
if res4 == 1
disp('Result 4 ...')
tic
Run_Connectivity;
toc
end

%% Result 5) Invariantly tuned unit responses enable invariant object detection
if res5 == 1
disp('Result 5 ...')
tic
Run_SVM;
toc
end