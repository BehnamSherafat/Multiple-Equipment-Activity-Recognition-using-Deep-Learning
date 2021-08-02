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

% ######################
num_act_eqp = [2, 2]; % Number of Activities of Each Equipment (e.g., 4 activities for CAT259D, 3 activities for CAT938M, and 4 acts for Jackhammer)
num_Classes = sum(num_act_eqp); 
num_total_combos = 9; % Total combination of signals for data augmentation
sample_Length  = 0.2*96000; % Length of signals to be mixed
per = 4; % period = 3 secs, Length of signals used for data augmentation
img_size = 224;
FFTSIZE = 512;
WINDOWSIZE = 418; 
noverlap = round(0.8*WINDOWSIZE);
% ######################

num_feat_stft = FFTSIZE/2+1;
hopSize = WINDOWSIZE - noverlap;
stft_length = fix((sample_Length-noverlap)/hopSize);

%% Load audio files of single-machine scenario

% ######################
[eqp{1}, fs1] = audioread("ExcavatorCAT329E_den.wav");% eqp_1 = CATMini (Hydraulic Excavator)
[eqp{2}, fs2] = audioread("LoaderCAT938K_den.wav");% eqp_2 = CAT938K (Loader)
% ######################

eqp{1} = eqp{1}(:,1);
eqp{2} = eqp{2}(:,1);

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
ampAdj = max(abs([eqp{1};eqp{2}]));
eqp{1} = eqp{1}/ampAdj;
eqp{2} = eqp{2}/ampAdj;

%% Selection of patterns (Sections of 0.5 seconds for training)

% ######################
% 1: Digging/Moving Arm/Turning
% 2: Dumping
signal{1} = [eqp{1}(14*fs:16*fs);eqp{1}(21.463*fs:23.465*fs)];
signal{2} = repmat([eqp{1}(27.245*fs:27.568*fs);eqp{1}(27.749*fs:28.169*fs);eqp{1}(28.486*fs:28.773*fs);eqp{1}(60.406*fs:60.996*fs);eqp{1}(107.152*fs:107.716*fs)],[2,1]);
% 1: Moving Forward/Moving Arm
% 2: Moving Backward/Moving Arm
signal{3} = [eqp{2}(30.614*fs:33.600*fs);eqp{2}(56.771*fs:58.408*fs)];
signal{4} = [eqp{2}(42.294*fs:43.580*fs);eqp{2}(44.087*fs:44.552*fs);eqp{2}(71.507*fs:73.805*fs)];
% ######################

signal = cellfun(@(x) x(1:per*fs),signal,'UniformOutput',false);
 
% for i=1:size(signal,2)
%     figure(i)
%     spectrogram(signal{i}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
%     plot(0:1/fs:(size(signal{i},1)/fs)-1/fs,signal{i})
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


% for i=1:size(signal,2)
%     figure(i+9)
%     spectrogram(signal{i}, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis')
% end

%% Arrange Signals

signals = cell(num_eqp, 1);

% ######################
signals{1}(1,:) = signal{1};
signals{1}(2,:) = signal{2};

signals{2}(1,:) = signal{3};
signals{2}(2,:) = signal{4};
% ######################

%%
tickets = 1:sample_Length:size(signals{1},2)-sample_Length+1;
num_tickets = numel(tickets);
combs = allcomb(1:num_act_eqp(1)*num_tickets,1:num_act_eqp(2)*num_tickets);

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
data_set = zeros(img_size, img_size, 1, 0);
num_iter = size(combs,1);
for m = 1:num_total_combos  
    for row = 1:num_iter
        sig_mixed = zeros(1, len_signals);
        data_labels_temp = zeros(num_Classes,1);
        max_sigs = zeros(1,num_eqp);
        sig = cell(num_eqp,1);
        for num_eqp = 1:size(combs,2)
            num_sign = ceil(combs(row, num_eqp)/num_tickets);
            card = tickets(combs(row, num_eqp) - floor(combs(row, num_eqp)/(num_tickets+0.000005))*num_tickets);
            sig{num_eqp,1} = signals{num_eqp}(num_sign,card:card+sample_Length-1);
            if num_eqp == 1
                vector = 1:num_act_eqp(num_eqp);
            else
                vector = num_act_eqp(1) + 1:num_act_eqp(1) + num_act_eqp(num_eqp);
            end
            str_new = num2str(vector(num_sign),'%d');
            data_labels_temp(str2double(str_new),:) = 1;
        end
        sig = cellfun(@(x) x/norm(x),sig,'UniformOutput',false);
        for j = 1:num_eqp
            max_sigs(1,j) = max(abs(sig{j}));
        end
        ampAdj = max(max_sigs);
        sig = cellfun(@(x) x/ampAdj,sig,'UniformOutput',false);
        
        
        
        %         for k = 1:num_eqp
        %             if m ==1
        %                 r = 1;
        %                 sig_mixed = sig_mixed+r.*sig{k};
        %             else
        %                 r = rand;
        %                 sig_mixed = sig_mixed+r.*sig{k};
        %             end
        %         end
        
        if m  == 1
            sig_mixed = mixing_sounds(1.*sig{1}, 1.*sig{2});
        elseif m == 2
            sig_mixed = mixing_sounds(0.1.*sig{1}, 1.*sig{2});
        elseif m == 3
            sig_mixed = mixing_sounds(0.2.*sig{1}, 0.8.*sig{2});
        elseif m == 4
            sig_mixed = mixing_sounds(0.3.*sig{1}, 0.7.*sig{2});
        elseif m == 5
            sig_mixed = mixing_sounds(0.4.*sig{1}, 0.6.*sig{2});
        elseif m == 6
            sig_mixed = mixing_sounds(0.6.*sig{1}, 0.4.*sig{2});
        elseif m == 7
            sig_mixed = mixing_sounds(0.7.*sig{1}, 0.3.*sig{2});
        elseif m == 8
            sig_mixed = mixing_sounds(0.8.*sig{1}, 0.2.*sig{2});
        elseif m == 9
            sig_mixed = mixing_sounds(1.*sig{1}, 0.1.*sig{2});
        end
        
        
        
        %         sig_mixed = sig_mixed./max(sig_mixed);
        
        spect = abs(spectrogram(sig_mixed, hann(WINDOWSIZE), noverlap, FFTSIZE, 'yaxis'));
        spect = imresize(spect,[img_size img_size],'method','bilinear','Antialiasing',true);
        %         imagesc(spect)
        data_set(:,:,:,(m-1)*num_iter+row) = spect;
        
        data_labels = cat(2, data_labels, data_labels_temp);
        disp(['Mix ', num2str((m-1)*num_iter+row), ' is created'])
    end
end

%% Take the log of the mix STFT. Normalize the values by their mean and standard deviation.

save('data_set_2Eqp.mat','data_set', '-v7.3');
save('data_labels_2Eqp.mat','data_labels', '-v7.3');

toc

function mixed = mixing_sounds(s1, s2)
%     mixed = s1 + s2; % Option 1
%     mixed = s1 + s2 - ((s1.*s2)/(2^16)); % Option 2

     % Option 3
    mixed = s1 + s2;
    s_max = 0.6*max(abs(mixed));
    pos_idx = (mixed > s_max);
    neg_idx = (mixed < -s_max);
    mixed(neg_idx) = -s_max;
    mixed(pos_idx) = +s_max;
    
    % Option 4
%     mixed = (s1 + s2);
%     ind_neg = (s1 < 0) & (s2 < 0);
%     mixed(ind_neg) = (s1(ind_neg) + s2(ind_neg)) - ((s1(ind_neg) .* s2(ind_neg))./-32768);
%     ind_pos = (s1 > 0) & (s2 > 0);
%     mixed(ind_pos) = (s1(ind_pos) + s2(ind_pos)) - ((s1(ind_pos) .* s2(ind_pos))./32768);
           
end




