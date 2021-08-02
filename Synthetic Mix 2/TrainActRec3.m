%% Multiple-equipment Activity Recognition using Deep Neural Network
% Author: Behnam Sherafat
% PhD Candiate at University of Utah

clc; clf; clear ; close all;
tic

%% Create Artificial Mix
% mixed_sig = zeros(size(eqp{1},1),1);
% for i = 1:num_eqp
%     mixed_sig = mixed_sig + eqp{i};
% end

%% Inputs

num_act_eqp = [4, 2, 2]; % Number of Activities of Each Equipment (e.g., 4 activities for CAT259D, 3 activities for CAT938M, and 4 acts for Jackhammer)

%% Load Pre-trained Network
load Trained_Net_2Eqp
layers = Trained_Net.Layers;

layers = [
    layers(1:12)
    fullyConnectedLayer(num_act_eqp(3),'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fc1')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','cl')];

layers(2).WeightLearnRateFactor = 0;
layers(2).BiasLearnRateFactor = 0;
layers(3).OffsetLearnRateFactor = 0;
layers(3).ScaleLearnRateFactor = 0;
layers(6).WeightLearnRateFactor = 0;
layers(6).BiasLearnRateFactor = 0;
layers(7).OffsetLearnRateFactor = 0;
layers(7).ScaleLearnRateFactor = 0;
layers(10).WeightLearnRateFactor = 0;
layers(10).BiasLearnRateFactor = 0;
layers(11).OffsetLearnRateFactor = 0;
layers(11).ScaleLearnRateFactor = 0;

%% Take the log of the mix STFT. Normalize the values by their mean and standard deviation.

load('data_set_2Eqp.mat','data_set');
load('data_labels_2Eqp.mat','data_labels');

%% Create Train and Test Data sets
% Cross varidation (train: 80%, validate: 10%, test: 10%)
% Shuffle DataSet
idx = randperm(size(data_set,4));
data_set = data_set(:,:,:,idx);
data_labels = data_labels(:,idx);

cv = cvpartition(size(data_set,4),'HoldOut',0.1);
idx = cv.test;

% Test Data
X_Test  = data_set(:,:,:,idx);
Y_Test  = data_labels(:,idx);

% Separate labels to train and val data
X_Train1 = data_set(:,:,:,~idx);
Y_Train1 = data_labels(:,~idx);

clear data_set_2Eqp
clear data_labels_2Eqp

cv = cvpartition(size(X_Train1,4),'HoldOut',0.1);
idx = cv.test;

% Validation Data
X_Val  = X_Train1(:,:,:,idx);
Y_Val  = Y_Train1(:,idx);

% Train Data
X_Train = X_Train1(:,:,:,~idx);
Y_Train = Y_Train1(:,~idx);

num_observartions = size(X_Train,4);

Y_Train1 = single(Y_Train(num_act_eqp(1)+num_act_eqp(2)+1:num_act_eqp(1)+num_act_eqp(2)+num_act_eqp(3),:));

for i =1:size(Y_Train1,2)
    Y_Train_re(1,i) = find(Y_Train1(:,i));
end

Y_Train_re = categorical(Y_Train_re);

Y_Val1 = single(Y_Val(num_act_eqp(1)+num_act_eqp(2)+1:num_act_eqp(1)+num_act_eqp(2)+num_act_eqp(3),:));

for i =1:size(Y_Val1,2)
    Y_Val_re(1,i) = find(Y_Val1(:,i));
end

Y_Val_re = categorical(Y_Val_re);

%% Specify the training options for the network
maxEpochs     = 1;
miniBatchSize = 300;
InitialLearnRate = 1e-3;
% Visualize the training progress in a plot

options = trainingOptions("adam", ...
    "InitialLearnRate", InitialLearnRate, ...
    "GradientDecayFactor", 0.9000, ...
    "SquaredGradientDecayFactor", 0.9900, ...
    "MaxEpochs",maxEpochs, ...
    "L2Regularization", 1.0000e-04, ...
    "MiniBatchSize",miniBatchSize, ...
    "SequenceLength","longest", ...
    "Shuffle","every-epoch",...
    "Verbose",0, ...
    "Plots","training-progress",...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropFactor",0.9, ...
    "LearnRateDropPeriod", 1, ...
    "ValidationFrequency",floor(size(X_Train,4)/miniBatchSize),...
    "ValidationData",{X_Val,Y_Val_re});

%%
% Train Deep Learning Network

Trained_Net = trainNetwork(X_Train,Y_Train_re,layers,options);

save('Trained_Net_2Eqp_3.mat','Trained_Net', '-v7.3');
 
% % Test Deep Learning Network on Test Data
% estimatedMasks0 = predict(Trained_Net,X_Test);
% 
% [~, q] = max(estimatedMasks0, [], 2) ;
% 
% Y_Pred_Test = zeros(num_act_eqp(2), size(q,1));
% 
% for i =1:size(q,1)
%     Y_Pred_Test(q(i),i) = 1;
% end
% 
% Y_Test1 = Y_Test(num_act_eqp(1)+1:num_act_eqp(1)+num_act_eqp(2),:);
% 
% figure(1)
% plotconfusion(Y_Pred_Test,Y_Test1)

