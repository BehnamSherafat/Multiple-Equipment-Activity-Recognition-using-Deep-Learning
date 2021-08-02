%% Multiple-equipment Activity Recognition using Deep Neural Network
% Author: Behnam Sherafat
% PhD Candiate at University of Utah
% 
clc; clf; clear ;
tic

%% Inputs
% ######################
num_act_eqp = [4, 3];
WindowLength  = 0.2*44100;
FFTSIZE = 512;
WINDOWSIZE = 192; 
noverlap = round(0.8*WINDOWSIZE);
img_size = 224;
time_per = 0.1;
sz_s = 22000;
% ######################

%% Test on Real Mixed Signal
load('Class_Dict.mat','M');

% ######################
load('Trained_Net_Eqp1_mobilenetv2.mat','Trained_Net');
% ######################


% Load Real Mix
ttt = 67; % 22 secs of mixed signal
[mixed_sig, fs] = audioread("RealMixed_den.wav");
mixed_sig = mixed_sig(1:ttt*fs);

T_test = size(mixed_sig,1)/fs;
mixed_sig = mixed_sig / norm(mixed_sig);
mixed_sig = mixed_sig / max(mixed_sig);
% Actual Activities
num_eqp = 2;
label_act = cell(num_eqp,1);
lps = fs;

% CATMiniHydraulicExcavator (Stop=1, Arm movement=2, Scraping/Loading=3, Dumping=4)

s1 = ceil((0)*lps+1);
f1 = ceil((8.9)*lps);
s2 = ceil((8.9)*lps);
f2 = ceil((9.9)*lps);
s3 = ceil((9.9)*lps);
f3 = ceil((35.721)*lps);
s4 = ceil((35.721)*lps);
f4 = ceil((40.227)*lps);
s5 = ceil((40.227)*lps);
f5 = ceil((48.706)*lps);
s6 = ceil((48.706)*lps);
f6 = ceil((51.308)*lps);
s7 = ceil((51.308)*lps);
f7 = ceil((60.771)*lps);
s8 = ceil((60.771)*lps);
f8 = ceil((61.118)*lps);
s9 = ceil((61.118)*lps);
f9 = ceil((63.8)*lps);
s10 = ceil((63.8)*lps);
f10 = ceil((64.8)*lps);
s11 = ceil((64.8)*lps);
f11 = ceil((T_test)*lps);

label_act{1}(s1:f1) = 1;
label_act{1}(s2:f2) = 4;
label_act{1}(s3:f3) = 1;
label_act{1}(s4:f4) = 2;
label_act{1}(s5:f5) = 1;
label_act{1}(s6:f6) = 4;
label_act{1}(s7:f7) = 1;
label_act{1}(s8:f8) = 3;
label_act{1}(s9:f9) = 2;
label_act{1}(s10:f10) = 3;
label_act{1}(s11:f11) = 2;
% ######################

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

subplot(3,1,1);
scatter(t,label_act{1},15,'filled','MarkerFaceColor',[0 0 0]);
title('Actual Labels for CAT 323F Excavator', 'FontSize', 20)
ax1 = gca;
set(ax1,'ytick',1:4)
set(ax1,'ylim',[1,4])
set(ax1,'yticklabel',{'Moving Forward/Backward','Moving Arm Up/Down','Scraping','Stop'},'FontSize',20)
xlim([0 T_test])

%%
nx = numel(mixed_sig);

ncol = floor((T_test*fs-WindowLength+1)/(time_per*fs))+1;
% xin(:,iCol,:) = x(1+hopSize*(iCol-1):nwin+hopSize*(iCol-1),:);

% fb = cwtfilterbank('SignalLength',WindowLength,'SamplingFrequency',fs,'FrequencyLimits',[0 10000]);
est_labels = zeros(1,ncol);

for m = 1:ncol
    windowed_sig = mixed_sig(round((m-1)*(time_per*fs)+1):min(length(mixed_sig), floor((m-1)*(time_per*fs)+WindowLength)));
    cwt_sig = abs(spectrogram(windowed_sig, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis'));
    cwt_sig = normalize(cwt_sig);
    cwt_sig = cat(3, cwt_sig, cwt_sig, cwt_sig);
    cwt_sig = imresize(cwt_sig,[img_size img_size],'method','bilinear','Antialiasing',true);
    %     imagesc(cwt_sig)
    out_label = predict(Trained_Net,cwt_sig);
    [p, q] = max(out_label, [], 2) ;
    est_labels(1,round((m-1)*(time_per*fs)+1):min(length(mixed_sig), floor((m-1)*(time_per*fs)+WindowLength))) = q;

end

est_labels1 = [est_labels est_labels(end)*ones(1, size(label_act{1},2)-size(est_labels,2))];

% for i = 1:num_eqp
%     subplot(3,1,i)
%     plot(est_labels(i,:))
%     axis tight
% end

subplot(3,1,2);
t1 = 0:1/fs:(size(est_labels1(1,:),2)/fs)-1/fs;
scatter(t1,est_labels1(1,:),8,'filled');
title('Predicted Labels for CAT 323F Excavator')
ax2 = gca;
set(ax2,'ytick',1:4)
set(ax2,'ylim',[1,4])
set(ax2,'yticklabel',{'Moving Forward/Backward','Moving Arm Up/Down','Scraping','Stop'},'FontSize', 20)
xlim([0 T_test])


cols = floor(size(est_labels1,2)/sz_s);

for m = 0:cols-1
    est_new = est_labels1(m*sz_s+1:m*sz_s+sz_s);
    filtered_sW(m*sz_s+1:m*sz_s+sz_s) = mode(est_new)*ones(1,sz_s);
end

filtered_sW1 = [filtered_sW filtered_sW(end)*ones(1, size(label_act{1},2)-size(filtered_sW,2))];

t1 = 0:1/fs:(size(filtered_sW1,2)/fs)-1/fs;
subplot(3,1,3);
scatter(t1,filtered_sW1,15,'filled','MarkerFaceColor',[0 1 0]);
ax3 = gca;
set(ax3,'ytick',1:4)
set(ax3,'ylim',[1,4])
set(ax3,'yticklabel',{'Moving Forward/Backward','Moving Arm Up/Down','Scraping','Stop'},'FontSize',20)
xlim([0 T_test])
title('Predicted Labels for CAT 323F Excavator', 'FontSize', 20)
xlabel('Time (s)', 'FontSize', 20)

% figure(2)
% plotconfusion(categorical(label_act{1}),categorical(est_labels1))
% 
figure(2)
plotconfusion(categorical(label_act{1}),categorical(filtered_sW1))
set(findobj(gca,'type','text'),'fontsize',20)


stats = confusionmatStats(categorical(label_act{1}), categorical(filtered_sW1));
stats.accuracy
stats.precision
stats.recall
stats.Fscore

toc