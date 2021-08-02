%% Multiple-equipment Activity Recognition using Deep Neural Network
% Author: Behnam Sherafat
% PhD Candiate at University of Utah

clc; clf; clear ;
tic

%% Inputs

num_act_eqp = [4, 3]; % Number of Activities of Each Equipment (e.g., 4 activities for CAT259D, 3 activities for CAT938M, and 4 acts for Jackhammer)
num_Classes = prod(num_act_eqp); % Number of Labels/Classes/Activities
WindowLength  = 0.2*44100;

FFTSIZE = 1024;
WINDOWSIZE = 1024;
noverlap = round(0.8*WINDOWSIZE);

%% Test on Real Mixed Signal
load('Class_Dict.mat','M');
load('Trained_Net_2Eqp_2.mat','Trained_Net');

% Load Real Mix
ttt = 16; % 22 secs of mixed signal
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

% CAT938K Loader (Stop=1, Moving Backward/Forward=2, Moving Arm Up/Down=3)

s0 = ceil((0)*lps+1);
f0 = ceil((2)*lps);
s1 = ceil((2)*lps+1);
f1 = ceil((7)*lps);
s2 = ceil((7)*lps);
f2 = ceil((11)*lps);
s3 = ceil((11)*lps);
f3 = ceil((12)*lps);
s4 = ceil((12)*lps);
f4 = ceil((13)*lps);
s5 = ceil((13)*lps);
f5 = ceil((15)*lps);
s6 = ceil((15)*lps);
f6 = ceil((T_test)*lps);

label_act{2}(s0:f0) = 3;
label_act{2}(s1:f1) = 2;
label_act{2}(s2:f2) = 3;
label_act{2}(s3:f3) = 2;
label_act{2}(s4:f4) = 3;
label_act{2}(s5:f5) = 2;
label_act{2}(s6:f6) = 1;

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
scatter(t,label_act{2},15,'filled','MarkerFaceColor',[0 0 0]);
title('Actual Labels for CAT 938K Loader', 'FontSize', 20)
ax1 = gca;
set(ax1,'ytick',1:3)
set(ax1,'ylim',[1,3])
set(ax1,'yticklabel',{'Stop','Moving','Moving Arm'}, 'FontSize', 20)
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
est_labels1 = [est_labels est_labels(end)*ones(1, size(label_act{2},2)-size(est_labels,2))];

% for i = 1:num_eqp
%     subplot(3,1,i)
%     plot(est_labels(i,:))
%     axis tight
% end

% subplot(3,1,2);
% t1 = 0:1/fs:(size(est_labels1(1,:),2)/fs)-1/fs;
% scatter(t1,est_labels1(1,:),8,'filled');
% title('Predicted Labels for CAT 938K Loader')
% ax2 = gca;
% set(ax2,'ytick',1:3)
% set(ax2,'ylim',[1,3])
% set(ax2,'yticklabel',{'Stop','Moving','Moving Arm'},'FontSize', 20)
% xlim([0 T_test])


sz_s = 44000;
cols = floor(size(est_labels1,2)/sz_s);

for m = 0:cols-1
    est_new = est_labels1(m*sz_s+1:m*sz_s+sz_s);
    filtered_sW(m*sz_s+1:m*sz_s+sz_s) = mode(est_new)*ones(1,sz_s);
end

filtered_sW1 = [filtered_sW filtered_sW(end)*ones(1, size(label_act{2},2)-size(filtered_sW,2))];

t1 = 0:1/fs:(size(filtered_sW1,2)/fs)-1/fs;
subplot(2,1,2);
scatter(t1,filtered_sW1,15,'filled','MarkerFaceColor',[0 1 0]);
ax3 = gca;
set(ax3,'ytick',1:3)
set(ax3,'ylim',[1,3])
set(ax3,'yticklabel',{'Stop','Moving','Moving Arm'},'FontSize',20)
xlim([0 T_test])
title('Predicted Labels for CAT 938K Loader','FontSize',20)
xlabel('Time (s)', 'FontSize', 20)

% figure(2)
% plotconfusion(categorical(label_act{2}),categorical(est_labels1))
% 
figure(2)
subplot(1,2,2)
plotconfusion(categorical(label_act{2}),categorical(filtered_sW1))
set(findobj(gca,'type','text'),'fontsize',20)
toc