%% Multiple-equipment Activity Recognition using Deep Neural Network
% Author: Behnam Sherafat
% PhD Candiate at University of Utah
% 
clc; clf; clear ; close all;
tic

%% Create Artificial Mix
% mixed_sig = zeros(size(eqp{1},1),1);
% for i = 1:num_eqp
%     mixed_sig = mixed_sig + eqp{i};
% end

%% Inputs
num_act_eqp = [4, 2, 2];
WindowLength  = 0.2*44100;

FFTSIZE = 1024;
WINDOWSIZE = 1024; 
noverlap = round(0.8*WINDOWSIZE);

%% Test on Real Mixed Signal
load('Class_Dict.mat','M');
load('Trained_Net_2Eqp.mat','Trained_Net');

% Load Real Mix
ttt = 20; % 22 secs of mixed signal
[sig1, fs1] = audioread("1.CAT259DExcavator_den.wav");
[sig2, fs2] = audioread("2.Jackhammer_den.wav");
[sig3, fs3] = audioread("3. SkyJackSJ6826Lift_den.wav");

fs = fs1;
sig1 = sig1(1:ttt*fs);
sig2 = sig2(1:ttt*fs);
sig3 = sig3(1:ttt*fs);

mixed_sig = sig1 + sig2 + sig3;

T_test = size(mixed_sig,1)/fs;

mixed_sig = mixed_sig / norm(mixed_sig);
mixed_sig = mixed_sig / max(mixed_sig);

%% Actual Activities
num_eqp = 3;
label_act = cell(num_eqp,1);
lps = fs;

% 1: Stop
% 2: Moving Forward
% 3: Arm/Shovel Movement
% 4: Moving Backward

s1 = ceil((0)*lps+1);
f1 = ceil((3.238)*lps);
s2 = ceil((3.238)*lps);
f2 = ceil((12.607)*lps);
s3 = ceil((12.607)*lps);
f3 = ceil((13.33)*lps);
s4 = ceil((13.33)*lps);
f4 = ceil((T_test)*lps);

label_act{1}(s1:f1) = 3;
label_act{1}(s2:f2) = 2;
label_act{1}(s3:f3) = 1;
label_act{1}(s4:f4) = 4;

%% Plot Actual Labels
t = 0:1/fs:T_test-1/fs;

figure(1)
% set(gca,'YDir','normal');
% set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
% set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
% title('CAT320E "Correct Label"',...
%     'FontSize', 28, 'FontWeight','bold');
% ylabel('Normalized Frequency', 'FontSize', 28, 'FontWeight','bold'); 
% xlabel('Time', 'FontSize', 28, 'FontWeight','bold');
% grid off
% % plot(t, label_act{1}, 'filled', 'Marker', 'none', 'Color', 'black');

subplot(2,1,1);
scatter(t,label_act{1},15,'filled','MarkerFaceColor',[1 0 0]);
title('Actual Labels for CAT 259D Compact Loader', 'FontSize', 20)
ax1 = gca;
set(ax1,'ytick',1:4)
set(ax1,'ylim',[1,4])
set(ax1,'yticklabel',{'Stop','Moving Forward','Arm/Shovel Movement','Moving Backward'}, 'FontSize', 20)
xlim([0 T_test])

%%
nx = numel(mixed_sig);
ncol = floor((T_test*fs-WindowLength+1)/(0.1*fs))+1;
% xin(:,iCol,:) = x(1+hopSize*(iCol-1):nwin+hopSize*(iCol-1),:);

% fb = cwtfilterbank('SignalLength',WindowLength,'SamplingFrequency',fs,'FrequencyLimits',[0 10000]);
est_labels = zeros(1,ncol);

for m = 1:ncol
%     windowed_sig = mixed_sig(round((m-1)*WindowLength+1):min(length(mixed_sig), floor((m-1)*WindowLength+WindowLength)));
    windowed_sig = mixed_sig(round((m-1)*(0.1*fs)+1):min(length(mixed_sig), floor((m-1)*(0.1*fs)+WindowLength)));
%     windowed_sig = windowed_sig / norm(windowed_sig);
%     windowed_sig = windowed_sig / max(windowed_sig);
%     spectrogram(windowed_sig, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
    cwt_sig = abs(spectrogram(windowed_sig, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis'));
    out_label = predict(Trained_Net,cwt_sig);
    [p, q] = max(out_label, [], 2) ;
    
%     testind = cellfun(@(x)isequal(x,q),values(M));
%     testkeys = keys(M);
%     msg_key = testkeys(testind);
%     
%     labels_new = split(msg_key,',');
%     for i = 1:num_eqp
%         est_labels(i,m) = str2num(labels_new{i});
%     end
%     est_labels(1,m) = str2num(labels_new{1});
    est_labels(1,round((m-1)*(0.1*fs)+1):min(length(mixed_sig), floor((m-1)*(0.1*fs)+WindowLength))) = q;

end

% est_labels1 = repelem(est_labels,WindowLength);
est_labels1 = [est_labels est_labels(end)*ones(1, size(label_act{1},2)-size(est_labels,2))];

% for i = 1:num_eqp
%     subplot(3,1,i)
%     plot(est_labels(i,:))
%     axis tight
% end

% subplot(3,1,2);
% t1 = 0:1/fs:(size(est_labels1(1,:),2)/fs)-1/fs;
% scatter(t1,est_labels1(1,:),8,'filled');
% title('Predicted Labels for CAT 323F Excavator')
% ax2 = gca;
% set(ax2,'ytick',1:4)
% set(ax2,'ylim',[1,4])
% set(ax2,'yticklabel',{'Moving Forward/Backward','Moving Arm Up/Down','Scraping','Stop'})
% xlim([0 T_test])

sz_s = 44000;
cols = floor(size(est_labels1,2)/sz_s);

for m = 0:cols-1
    est_new = est_labels1(m*sz_s+1:m*sz_s+sz_s);
    filtered_sW(m*sz_s+1:m*sz_s+sz_s) = mode(est_new)*ones(1,sz_s);
end

filtered_sW1 = [filtered_sW filtered_sW(end)*ones(1, size(label_act{1},2)-size(filtered_sW,2))];

t1 = 0:1/fs:(size(filtered_sW1,2)/fs)-1/fs;
subplot(2,1,2);
scatter(t1,filtered_sW1,15,'filled','MarkerFaceColor',[0 0 1]);
ax3 = gca;
set(ax3,'ytick',1:4)
set(ax3,'ylim',[1,4])
set(ax3,'yticklabel',{'Stop','Moving Forward','Arm/Shovel Movement','Moving Backward'},'FontSize',20)
xlim([0 T_test])
title('Predicted Labels for CAT 259D Compact Loader','FontSize',20)
xlabel('Time (s)', 'FontSize', 20)

% figure(2)
% plotconfusion(categorical(label_act{1}),categorical(est_labels1))
% 
figure(2)
plotconfusion(categorical(label_act{1}),categorical(filtered_sW1))
set(findobj(gca,'type','text'),'fontsize',20)
toc