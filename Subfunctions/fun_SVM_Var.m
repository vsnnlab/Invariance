function [SVM_face_multi] = fun_SVM_Var(net_rand,num_cell,IND_Face,layersSet,indLayer,numNeuron,var_type, target_class, train_view, test_view)

numIMGtot = 120;
[IMG_center, ~] = fun_GetStim('SVM_var', var_type, target_class, strcat(num2str(train_view), '.0')); 
[IMG_var, ~] = fun_GetStim('SVM_var', var_type, target_class,  strcat(num2str(test_view), '.0')); 
IMG_center = single(IMG_center);
IMG_var = single(IMG_var);

%% SVM
ratio_Tr_Te = 3;
if numNeuron == 43264
    reN = 1;
else
    reN = 10;
end

% split train / test
iFace = [1:numIMGtot/2]; iObject = [numIMGtot/2+1:numIMGtot]; iObject = iObject(randperm(length(iObject)));
iFaceTrain = iFace(randperm(numIMGtot/2,numIMGtot/2/ratio_Tr_Te*2)); iFaceTest = setdiff(iFace,iFaceTrain);
iObjectTrain = iObject(randperm(numIMGtot/2,numIMGtot/2/ratio_Tr_Te*2)); iObjectTest = setdiff(iObject,iObjectTrain);

indTrain = [iFaceTrain';iObjectTrain']; indTest = [iFaceTest';iObjectTest'];
YTrain = [ones(numIMGtot/2/ratio_Tr_Te*2,1); zeros(numIMGtot/2/ratio_Tr_Te*2,1)];
YTest = [ones(numIMGtot/2/ratio_Tr_Te,1); zeros(numIMGtot/2/ratio_Tr_Te,1)];

% response
act_rand_center = activations(net_rand,IMG_center,layersSet{indLayer});
act_center = reshape(act_rand_center,num_cell,size(IMG_center,4));
act_rand_var = activations(net_rand,IMG_var,layersSet{indLayer});
act_var = reshape(act_rand_var,num_cell,size(IMG_var,4));

XTrain1 = act_center(IND_Face,indTrain); XTest1 = act_var(IND_Face,indTest);

%% Multiple neuron SVM
SVM_face_multi_array = zeros(reN:1);
for ii=1:reN
    ind_f = randperm(length(IND_Face),numNeuron);
    Mdl = fitcecoc(XTrain1(ind_f,:)',YTrain);
    YPredict = predict(Mdl,XTest1(ind_f,:)');
    SVM_face_multi_array = length(find(YTest == YPredict))./length(YTest);
end
SVM_face_multi = nanmean(SVM_face_multi_array);
end