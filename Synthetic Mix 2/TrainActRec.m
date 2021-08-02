%% Multiple-equipment Activity Recognition using Deep Neural Network
% Author: Behnam Sherafat
% PhD Candiate at University of Utah
% 
% clc; clf; clear ; close all;
% tic

%% Inputs
num_act_eqp = [4, 2, 2]; % Number of Activities of Each Equipment (e.g., 4 activities for CAT259D, 3 activities for CAT938M, and 4 acts for Jackhammer)
sample_Length  = 0.2*44100; % Length of signals to be mixed
FFTSIZE = 1024;
WINDOWSIZE = 1024; 
noverlap = round(0.8*WINDOWSIZE);
num_feat_stft = FFTSIZE/2+1;
hopSize = WINDOWSIZE - noverlap;
stft_length = fix((sample_Length-noverlap)/hopSize);

%% Take the log of the mix STFT. Normalize the values by their mean and standard deviation.

% load('data_set_2Eqp.mat','data_set');
% load('data_labels_2Eqp.mat','data_labels');

% data_set = normalize(data_set);

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

clear data_set_4Eqp
clear data_labels_4Eqp

cv = cvpartition(size(X_Train1,4),'HoldOut',0.1);
idx = cv.test;

% Validation Data
X_Val  = X_Train1(:,:,:,idx);
Y_Val  = Y_Train1(:,idx);

% Train Data
X_Train = X_Train1(:,:,:,~idx);
Y_Train = Y_Train1(:,~idx);

num_observartions = size(X_Train,4);

Y_Train1 = single(Y_Train(1:num_act_eqp(1),:));

for i =1:size(Y_Train1,2)
    Y_Train_re(1,i) = find(Y_Train1(:,i));
end

Y_Train_re = categorical(Y_Train_re);

Y_Val1 = single(Y_Val(1:num_act_eqp(1),:));

for i =1:size(Y_Val1,2)
    Y_Val_re(1,i) = find(Y_Val1(:,i));
end

Y_Val_re = categorical(Y_Val_re);

%% Define Deep Learning Model

layers = [
    imageInputLayer([num_feat_stft stft_length 1],'Normalization','none','Name','in')
    
    convolution2dLayer(3,64,'Padding','same','Stride',[4 2],'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    leakyReluLayer('Name','lrelu1')
    
    maxPooling2dLayer(2,'Stride',1,'Name','max1')
    
    convolution2dLayer(3,128,'Padding','same','Stride',[4 2],'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer('Name','lrelu2')
    
    maxPooling2dLayer(2,'Stride',1,'Name','max2')
    
    convolution2dLayer(3,32,'Padding','same', 'Stride', [4 1],'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer('Name','lrelu3')
    
    fullyConnectedLayer(num_act_eqp(1),'Name','fc1')
    dropoutLayer(0.1)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','cl')];

%% Specify the training options for the network
maxEpochs     = 3;
miniBatchSize = 300;

% Visualize the training progress in a plot

options = trainingOptions("adam", ...
    "InitialLearnRate", 3e-3, ...
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
    

%% Train Deep Learning Network
%
Trained_Net = trainNetwork(X_Train,Y_Train_re,layers,options);

save('Trained_Net_2Eqp.mat','Trained_Net', '-v7.3');
%  
% % Test Deep Learning Network on Test Data
% estimatedMasks0 = predict(Trained_Net,X_Test);
% 
% [~, q] = max(estimatedMasks0, [], 2) ;
% 
% Y_Pred_Test = zeros(num_act_eqp(1), size(q,1));
% 
% for i =1:size(q,1)
%     Y_Pred_Test(q(i),i) = 1;
% end
% 
% Y_Test1 = Y_Test(1:4,:);
% 
% figure(1)
% plotconfusion(Y_Pred_Test,Y_Test1)
% 
% %%
% % Test Deep Learning Network on Train Data
% estimatedMasks1 = predict(Trained_Net,X_Train);
% 
% [~, q] = max(estimatedMasks1, [], 2) ;
% 
% Y_Pred_Train = zeros(num_act_eqp(1), size(q,1));
% 
% for i =1:size(q,1)
%     Y_Pred_Train(q(i),i) = 1;
% end
% 
% Y_Train1 = Y_Train(1:4,:);
% 
% figure(2)
% plotconfusion(Y_Pred_Train,Y_Train1)
% 
% %%
% % Test Deep Learning Network on Val Data
% estimatedMasks2 = predict(Trained_Net,X_Val);
% 
% [~, q] = max(estimatedMasks2, [], 2) ;
% 
% Y_Pred_Val = zeros(num_act_eqp(1), size(q,1));
% 
% for i =1:size(q,1)
%     Y_Pred_Val(q(i),i) = 1;
% end
% 
% Y_Val1 = Y_Val(1:4,:);
% 
% figure(3)
% plotconfusion(Y_Pred_Val,Y_Val1)
