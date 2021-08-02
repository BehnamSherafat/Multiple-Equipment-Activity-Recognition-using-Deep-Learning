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
num_act_eqp = [2, 2];
WindowLength  = 0.2*96000;

FFTSIZE = 2048;
WINDOWSIZE = 2048; 
noverlap = round(0.8*WINDOWSIZE);

% num_feat_stft = FFTSIZE/2+1;
% 
% hopSize = WINDOWSIZE - noverlap;
% stft_length = fix((sample_Length-noverlap)/hopSize);

%% Test on Real Mixed Signal
load('Class_Dict.mat','M');
load('Trained_Net_2Eqp.mat','Trained_Net');

% Load Real Mix
ttt = 25; % 22 secs of mixed signal
[mixed_sig, fs] = audioread("RealMixed_den.wav");
mixed_sig = mixed_sig(1:ttt*fs);

%% Downsampling (fs1 = 44100 ==> fs2 = 441)
% down_int = 100;
% fs = fs3/down_int;
% 
% mixed_sig = downsample(mixed_sig,down_int);
% mixed_sig = mixed_sig(1:floor(size(mixed_sig,1)/noverlap)*noverlap);
T_test = size(mixed_sig,1)/fs;

mixed_sig = mixed_sig / norm(mixed_sig);
mixed_sig = mixed_sig / max(mixed_sig);

%% Actual Activities
num_eqp = 2;
label_act = cell(num_eqp,1);
lps = fs;

% CATMiniHydraulicExcavator (Stop=1, Arm movement=2, Scraping/Loading=3, Dumping=4)

s1 = ceil((0)*lps+1);
f1 = ceil((7.5)*lps);
s2 = ceil((7.5)*lps);
f2 = ceil((14.00)*lps);
s3 = ceil((14.00)*lps);
f3 = ceil((17)*lps);
s4 = ceil((17)*lps);
f4 = ceil((T_test)*lps);
% s7 = ceil((27.597)*lps);
% f7 = ceil((38.151)*lps);
% s8 = ceil((38.151)*lps);
% f8 = ceil((48.369)*lps);
% s9 = ceil((48.369)*lps);
% f9 = ceil((51.872)*lps);
% s10 = ceil((51.872)*lps);
% f10 = ceil((58.975)*lps);
% s11 = ceil((58.975)*lps);
% f11 = ceil((T_test)*lps);
% s12 = ceil((58.241)*lps);
% f12 = ceil((4.438)*lps);
% s13 = ceil((4.438)*lps);
% f13 = ceil((4.442)*lps);
% s14 = ceil((4.442)*lps);
% f14 = ceil((7.155)*lps);
% s15 = ceil((7.155)*lps);
% f15 = ceil((7.453)*lps);
% s16 = ceil((7.453)*lps);
% f16 = ceil((8.011)*lps);
% s17 = ceil((8.011)*lps);
% f17 = ceil((12.734)*lps);
% s18 = ceil((12.734)*lps);
% f18 = ceil((13)*lps);
% s19 = ceil((13)*lps);
% f19 = ceil((14.38)*lps);
% s20 = ceil((14.38)*lps);
% f20 = ceil((15.234)*lps);
% s21 = ceil((15.234)*lps);
% f21 = ceil((15.67)*lps);
% s22 = ceil((15.67)*lps);
% f22 = ceil((19.416)*lps);
% s23 = ceil((19.416)*lps);
% f12 = ceil((T_test)*lps);

label_act{1}(s1:f1) = 1;
label_act{1}(s2:f2) = 1;
label_act{1}(s3:f3) = 2;
label_act{1}(s4:f4) = 1;
% label_act{1}(s5:f5) = 2;
% label_act{1}(s6:f6) = 2;
% label_act{1}(s7:f7) = 1;
% label_act{1}(s8:f8) = 2;
% label_act{1}(s9:f9) = 3;
% label_act{1}(s10:f10) = 2;
% label_act{1}(s11:f11) = 1;
% label_act{1}(s12:f12) = 3;
% label_act{1}(s13:f13) = 1;
% label_act{1}(s14:f14) = 2;
% label_act{1}(s15:f15) = 3;
% label_act{1}(s16:f16) = 2;
% label_act{1}(s17:f17) = 3;
% label_act{1}(s18:f18) = 2;
% label_act{1}(s19:f19) = 3;
% label_act{1}(s20:f20) = 2;
% label_act{1}(s21:f21) = 3;
% label_act{1}(s22:f22) = 2;
% label_act{1}(s23:f23) = 3;

% label_act{1} = label_act{1}(1:9408);
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
title('Actual Labels for CAT 329E Excavator', 'FontSize', 20)
ax1 = gca;
set(ax1,'ytick',1:2)
set(ax1,'ylim',[1,2])
set(ax1,'yticklabel',{'Digging/Moving Arm','Dumping'}, 'FontSize', 20)
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
% title('Predicted Labels for Excavator CAT329E')
% ax2 = gca;
% set(ax2,'ytick',1:2)
% set(ax2,'ylim',[1,2])
% set(ax2,'yticklabel',{'Digging/Moving Arm','Dumping'})
% xlim([0 T_test])

sz_s = 88000;
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
set(ax3,'ytick',1:2)
set(ax3,'ylim',[1,2])
set(ax3,'yticklabel',{'Digging/Moving Arm','Dumping'},'FontSize',20)
xlim([0 T_test])
title('Predicted Labels for CAT 329E Excavator','FontSize',20)
xlabel('Time (s)', 'FontSize', 20)

% figure(2)
% plotconfusion(categorical(label_act{1}),categorical(est_labels1))
% 
figure(2)
plotconfusion(categorical(label_act{1}),categorical(filtered_sW1))
set(findobj(gca,'type','text'),'fontsize',20)
toc