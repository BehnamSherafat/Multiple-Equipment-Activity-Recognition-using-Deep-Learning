%% Multiple-equipment Activity Recognition using Deep Neural Network
% Author: Behnam Sherafat
% PhD Candiate at University of Utah

clc; clf; clear ; close all;
tic

%% Audio Denoising: method = 'mcra2'; 
% [A_ori, fsA] = audioread('RealMixed.wav');
% A_ori(A_ori==0)=10^-4;
% A_ori = A_ori(:, 1);
% audiowrite('RealMixed_ori.wav', A_ori, fsA);
% % 
% % % Audio Denoising: method = 'mcra2'; 
% specsub_ns('RealMixed_ori.wav', 'mcra2', 'RealMixed_den.wav')

%% Inputs

num_act_eqp = [4, 2, 2]; % Number of Activities of Each Equipment (e.g., 4 activities for CAT259D, 3 activities for CAT938M, and 4 acts for Jackhammer)
num_Classes = sum(num_act_eqp); 
num_total_combos = 2; % Total combination of signals for data augmentation

sample_Length  = 0.2*44100; % Length of signals to be mixed
per = 4; % period = 3 secs, Length of signals used for data augmentation

FFTSIZE = 1024;
WINDOWSIZE = 1024; 
noverlap = round(0.8*WINDOWSIZE);

num_feat_stft = FFTSIZE/2+1;

hopSize = WINDOWSIZE - noverlap;
stft_length = fix((sample_Length-noverlap)/hopSize);

%% Load audio files of single-machine scenario

[eqp{1}, fs1] = audioread("1.CAT259DExcavator_den.wav");
[eqp{2}, fs2] = audioread("2.Jackhammer_den.wav");
[eqp{3}, fs3] = audioread("3. SkyJackSJ6826Lift_den.wav");

eqp{1} = eqp{1}(:,1);
eqp{2} = eqp{2}(:,1);
eqp{3} = eqp{3}(:,1);

fs = fs1;
num_eqp = size(eqp, 2);

%% Plot Signals
% figure(1)
% subplot(2,1,1)
% plot(0:1/fs:(size(eqp{2},1)/fs)-(1/fs),eqp{2})
% xlabel('Time (s)') 
% ylabel('Amplitutde')
% title('Original')
% axis tight
% 
% subplot(2,1,2)
% plot(0:1/fs:(size(eqp{1},1)/fs)-(1/fs),eqp{1})
% xlabel('Time (s)') 
% ylabel('Amplitutde')
% title('Denoised')
% axis tight
% 
% 
% figure(2)
% subplot(2,1,1)
% spectrogram(eqp{2}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
% xlabel('Time (s)') 
% title('Original')
% axis tight
% 
% subplot(2,1,2)
% spectrogram(eqp{1}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
% xlabel('Time (s)') 
% title('Denoised')
% axis tight

%% Scale the signals to the same power

eqp{1} = eqp{1}/norm(eqp{1});
eqp{2} = eqp{2}/norm(eqp{2});
eqp{3} = eqp{3}/norm(eqp{3});
ampAdj = max(abs([eqp{1};eqp{2};eqp{3}]));
eqp{1} = eqp{1}/ampAdj;
eqp{2} = eqp{2}/ampAdj;
eqp{3} = eqp{3}/ampAdj;

%% Selection of patterns (Sections of 0.5 seconds for training)

% cwt(signal{2},'FilterBank',cwtfilterbank('SignalLength',size(signal{2},1),'SamplingFrequency',fs,'FrequencyLimits',[0 10000]))

% 1: Stop
% 2: Moving Forward
% 3: Arm/Shovel Movement
% 4: Moving Backward
signal{1} = repmat([eqp{1}(12.607*fs:13.33*fs)],[60,1]);
signal{2} = [eqp{1}(4*fs:8*fs)];
signal{3} = repmat([eqp{1}(0.1*fs:3*fs)],[60,1]);
signal{4} = [eqp{1}(15*fs:19*fs)];

% 1: Moving Arm
% 2: Drilling
signal{5} = [eqp{2}(8.1*fs:12.1*fs)];
signal{6} = [eqp{2}(2*fs:6*fs)];

% 1: Stop
% 2: Lifting Up
signal{7} = [eqp{3}(3*fs:7*fs)];
signal{8} = [eqp{3}(13*fs:17*fs)];

signal = cellfun(@(x) x(1:per*fs),signal,'UniformOutput',false);

% for i=1:size(signal,2)
%     figure(i)
%     spectrogram(signal{i}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
% end

%% Scale the signals to the same power

max_sig = zeros(1,num_eqp);
num_sig = size(signal, 2);

signal = cellfun(@(x) x/norm(x),signal,'UniformOutput',false);

for i = 1:num_sig
    max_sig(1,i) = max(abs(signal{i}));
end

ampAdj = max(max_sig);
signal = cellfun(@(x) x/ampAdj,signal,'UniformOutput',false);

% 
% for i=1:size(signal,2)
%     figure(i+9)
%     spectrogram(signal{i}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
% end

%% Arrange Signals

signals = cell(num_eqp, 1);

signals{1}(1,:) = signal{1};
signals{1}(2,:) = signal{2};
signals{1}(3,:) = signal{3};
signals{1}(4,:) = signal{4};

signals{2}(1,:) = signal{5};
signals{2}(2,:) = signal{6};

signals{3}(1,:) = signal{7};
signals{3}(2,:) = signal{8};

%%
tickets = 1:sample_Length:size(signals{1},2)-sample_Length+1;
num_tickets = numel(tickets);
% num_tickets = 1;
combs = allcomb(1:num_act_eqp(1)*num_tickets,1:num_act_eqp(2)*num_tickets,1:num_act_eqp(3)*num_tickets);

%% Create Dictionary of Samples and Labels

act_eqp = {};
for k = 1:num_eqp
    if k ==1
        ind_start = 1;
        ind_end = ind_start + num_act_eqp(k) - 1;
    else
        ind_start = ind_end + 1;
        ind_end = ind_start + num_act_eqp(k) - 1;
    end  
    act_eqp{k} = ind_start:ind_end;
end

dict_labels = combvec(act_eqp{:})';
dict_labels = string(dict_labels);
dict_labels = join(dict_labels,",");

dict_vals = [];
for i = 1:prod(num_act_eqp)
    dict_vals = [dict_vals i];
    i = i + 1;
end

M = containers.Map(dict_labels,dict_vals);
save('Class_Dict.mat','M', '-v7.3');

%%

% fb = cwtfilterbank('SignalLength',sample_Length,'SamplingFrequency',fs,'FrequencyLimits',[0 10000]);

len_signals = sample_Length;
data_labels = zeros(num_Classes,0);
data_set = zeros(num_feat_stft, stft_length, 1, 0);
jjj = 1;
for m = 1:num_total_combos  
    for row = 1:size(combs,1)
        sig_mixed = zeros(1, len_signals);
        data_labels_temp = zeros(num_Classes,1);
        str = '';
        max_sigs = zeros(1,num_eqp);
        sig = cell(num_eqp,1);
        for col = 1:size(combs,2)
            num_sig = ceil(combs(row, col)/num_tickets);
            card = tickets(combs(row, col) - floor(combs(row, col)/(num_tickets+0.000005))*num_tickets);
            sig{col,1} = signals{col}(num_sig,card:card+sample_Length-1);
%             sig{col,1} = signals{col}(num_sig,:);
            if col ==1
                ind_start = 1;
                ind_end = ind_start + num_act_eqp(col) - 1;
            else
                ind_start = ind_end + 1;
                ind_end = ind_start + num_act_eqp(col) - 1;
            end
            vector = ind_start:ind_end;
            str_new = num2str(vector(num_sig),'%d');
            data_labels_temp(str2double(str_new),:) = 1;
            if (str == "")
                str = append(str,str_new);
            else
                str = append(str,',',str_new);
            end
        end
        sig = cellfun(@(x) x/norm(x),sig,'UniformOutput',false);
        for j = 1:num_eqp
            max_sigs(1,j) = max(abs(sig{j}));
        end
        ampAdj = max(max_sigs);
        sig = cellfun(@(x) x/ampAdj,sig,'UniformOutput',false);
        
        for k = 1:num_eqp
            if m ==1
                r = 1;
                sig_mixed = sig_mixed+r*sig{k};
            else
                r = rand;
%                 r = 1;
                sig_mixed = sig_mixed+r*sig{k};
%                 sig_mixed = sig_mixed+r*awgn(sig{k},10,'measured','linear');
            end
        end
        
        g = 0.5;
%         g = 1;
        sig_mixed = g * sig_mixed;
        
%         figure(1)
%         spectrogram(sig{1}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
%         figure(2)
%         spectrogram(sig{2}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
%         figure(3)
%         spectrogram(sig_mixed, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
%         close all
        
        data_set(:,:,:,jjj) = abs(spectrogram(sig_mixed, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis'));
        data_labels = cat(2, data_labels, data_labels_temp);
        disp(['Mix ', num2str(jjj), ' is created'])
        jjj = jjj + 1;
    end
end

%% Take the log of the mix STFT. Normalize the values by their mean and standard deviation.

save('data_set_2Eqp.mat','data_set', '-v7.3');
save('data_labels_2Eqp.mat','data_labels', '-v7.3');

toc