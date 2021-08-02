%% Multiple-equipment Activity Recognition using Deep Neural Network
% Author: Behnam Sherafat
% PhD Candiate at University of Utah
% 
clc; clf; clear ; close all;
% delete(gcp('nocreate'))
tic

%% Inputs
% ######################
num_act_eqp = [4, 3]; % Number of Activities of Each Equipment (e.g., 4 activities for CAT259D, 3 activities for CAT938M, and 4 acts for Jackhammer)
img_size = 224;
maxEpochs = 3;
miniBatchSize = 50;
L2Regularization = 1.0000e-04;
XTranslation = 100;
% ######################

%% Take the log of the mix STFT. Normalize the values by their mean and standard deviation.

load('data_set_2Eqp.mat','data_set');
load('data_labels_2Eqp.mat','data_labels');

% data_set = normalize(data_set);

%% Create Train and Test Data sets
% Cross varidation (train: 80%, validate: 10%, test: 10%)
% Shuffle DataSet
idx = randperm(size(data_set,4));
X_Train = data_set(:,:,:,idx);
Y_Train = data_labels(:,idx);

clear data_set
% Train Labels
Y_Train1 = single(Y_Train(1:num_act_eqp(1),:));

for i =1:size(Y_Train1,2)
    Y_Train_re(1,i) = find(Y_Train1(:,i));
end

Y_Train_re = categorical(Y_Train_re);

%% Make Input Comaptible with the input Size of Transfer Learning
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',[-XTranslation XTranslation]);

auimdsTrain = augmentedImageDatastore([img_size img_size 3],X_Train,Y_Train_re,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);

%% Transfer Learning using AlexNet, SqueezeNet, ShuffleNet, etc.
% analyzeNetwork(net)
numClasses = num_act_eqp(1);

% ######################
%======> resnet18
% net = resnet18;
% layers = layerGraph(net); 
% 
% newLearnableLayer = fullyConnectedLayer(numClasses, ...
%     'Name','new_fc', ...
%     'WeightLearnRateFactor',20, ...
%     'BiasLearnRateFactor',20);
% layers = replaceLayer(layers,'fc1000',newLearnableLayer);
% newClassLayer = classificationLayer('Name','new_classoutput');
% layers = replaceLayer(layers,'ClassificationLayer_predictions',newClassLayer);

%======> MobileNet-v2
net = mobilenetv2;
layers = layerGraph(net); 
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',0.9, ...
    'BiasLearnRateFactor',1);
layers = replaceLayer(layers,'Logits',newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
layers = replaceLayer(layers,'ClassificationLayer_Logits',newClassLayer);

%======> SqueezeNet
% net = squeezenet;
% lgraph = layerGraph(net);
% % [learnableLayer,classLayer] = findLayersToReplace(lgraph);
% newConvLayer =  convolution2dLayer([1, 1],numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
% lgraph = replaceLayer(lgraph,'conv10',newConvLayer);
% newClassificatonLayer = classificationLayer('Name','new_classoutput');
% lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificatonLayer);

%======> nasnetlarge
% net = nasnetmobile;
% layers = layerGraph(net); 
% newLearnableLayer = fullyConnectedLayer(numClasses, ...
%     'Name','new_fc', ...
%     'WeightLearnRateFactor',20, ...
%     'BiasLearnRateFactor',20);
% layers = replaceLayer(layers,'predictions',newLearnableLayer);
% newClassLayer = classificationLayer('Name','new_classoutput');
% layers = replaceLayer(layers,'ClassificationLayer_predictions',newClassLayer);

%======> googleNet
% net = googlenet;
% layers = layerGraph(net); 
% newLearnableLayer = fullyConnectedLayer(numClasses, ...
%     'Name','new_fc', ...
%     'WeightLearnRateFactor',0.9, ...
%     'BiasLearnRateFactor',1);
% layers = replaceLayer(layers,'loss3-classifier',newLearnableLayer);
% newClassLayer = classificationLayer('Name','new_classoutput');
% layers = replaceLayer(layers,'output',newClassLayer);

% ######################
%% Model Training Options

% Best Options for AlexNet

% options = trainingOptions("adam", ...
%     'ExecutionEnvironment','parallel', ...
%     "InitialLearnRate", 3e-3, ...
%     "GradientDecayFactor", 0.9000, ...
%     "SquaredGradientDecayFactor", 0.9900, ...
%     "MaxEpochs",maxEpochs, ...
%     "L2Regularization", 1.0000e-05, ...
%     "MiniBatchSize",miniBatchSize, ...
%     "SequenceLength","longest", ...
%     "Shuffle","every-epoch",...
%     "Verbose",0, ...
%     "Plots","training-progress",...
%     "LearnRateSchedule","piecewise",...
%     "LearnRateDropFactor",0.9, ...
%     "LearnRateDropPeriod", 1);

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','parallel', ...
    'MiniBatchSize',miniBatchSize, ...
    "L2Regularization", L2Regularization, ...
    'MaxEpochs',maxEpochs, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train Deep Learning Network
clear X_Train
clear Y_Train_re
toc
Trained_Net = trainNetwork(auimdsTrain,layers,options);

save('Trained_Net_Eqp1_mobilenetv2.mat','Trained_Net', '-v7.3');

% I = deepDreamImage(Trained_Net,Trained_Net.Layers(69).Name,[1 2 3 4], ...
%     'Verbose',false, ...
%     'NumIterations',200, ...
%     'PyramidLevels',3);
% figure
% I = imtile(I,'ThumbnailSize',[250 250]);
% imshow(I)