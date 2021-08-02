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
sz_s = 44000;
% ######################

%% Test on Real Mixed Signal
load('Class_Dict.mat','M');

% ######################
load('Trained_Net_Eqp1_Resnet18.mat','Trained_Net');
% ######################


% Load Real Mix
ttt = 22; % 22 secs of mixed signal
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
f1 = ceil((0.190)*lps);
s2 = ceil((0.190)*lps);
f2 = ceil((4.438)*lps);
s13 = ceil((4.438)*lps);
f13 = ceil((4.7)*lps);
s14 = ceil((4.7)*lps);
f14 = ceil((7.155)*lps);
s15 = ceil((7.155)*lps);
f15 = ceil((7.453)*lps);
s16 = ceil((7.453)*lps);
f16 = ceil((8.011)*lps);
s17 = ceil((8.011)*lps);
f17 = ceil((12.25)*lps);
s18 = ceil((12.25)*lps);
f18 = ceil((12.7)*lps);
s26 = ceil((12.7)*lps);
f26 = ceil((12.9)*lps);
s27 = ceil((12.9)*lps);
f27 = ceil((13.6)*lps);
s19 = ceil((13.6)*lps);
f19 = ceil((14.38)*lps);
s20 = ceil((14.38)*lps);
f20 = ceil((14.725)*lps);
s21 = ceil((14.725)*lps);
f21 = ceil((14.866)*lps);
s22 = ceil((14.866)*lps);
f22 = ceil((15.234)*lps);
s23 = ceil((15.234)*lps);
f23 = ceil((15.67)*lps);
s24 = ceil((15.67)*lps);
f24 = ceil((19.416)*lps);
s25 = ceil((19.416)*lps);
f25 = ceil((T_test)*lps);

label_act{1}(s1:f1) = 2;
label_act{1}(s2:f2) = 3;
label_act{1}(s13:f13) = 3;
label_act{1}(s14:f14) = 2;
label_act{1}(s15:f15) = 2;
label_act{1}(s16:f16) = 2;
label_act{1}(s17:f17) = 3;
label_act{1}(s18:f18) = 2;
label_act{1}(s19:f19) = 3;
label_act{1}(s20:f20) = 2;
label_act{1}(s21:f21) = 2;
label_act{1}(s22:f22) = 2;
label_act{1}(s23:f23) = 2;
label_act{1}(s24:f24) = 2;
label_act{1}(s25:f25) = 2;
label_act{1}(s26:f26) = 1;
label_act{1}(s27:f27) = 2;
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
title('Actual Labels for CAT Mini Hydraulic Excavator', 'FontSize', 20)
ax1 = gca;
set(ax1,'ytick',1:4)
set(ax1,'ylim',[1,4])
set(ax1,'yticklabel',{'Stop','Arm movement','Scraping/Loading','Dumping'},'FontSize',20)
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
title('Predicted Labels for CATMini Hydraulic Excavator')
ax2 = gca;
set(ax2,'ytick',1:4)
set(ax2,'ylim',[1,4])
set(ax2,'yticklabel',{'Stop','Arm movement','Scraping/Loading','Dumping'},'FontSize', 20)
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
set(ax3,'yticklabel',{'Stop','Arm movement','Scraping/Loading','Dumping'},'FontSize',20)
xlim([0 T_test])
title('Predicted Labels for CAT Mini Hydraulic Excavator', 'FontSize', 20)
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