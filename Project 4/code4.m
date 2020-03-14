%% convert files to .mat format
clear; close all; clc

filenames = {'beethoven-moonlightsonata.mp3', 'Deflo & Lliam Taylor - Spotlight (feat. AWA).mp3', 'divine-joed.mp3','peculate-valeoftears.mp3','romanJewels.mp3','sadpuppy-youhavetheblame.mp3'};


filename = filenames{1};
[y1,Fs1] = audioread(filename);
y1 = mean(y1,2);
save m1 y1 Fs1
clear filename

filename = filenames{2};
[y2,Fs2] = audioread(filename);
y2 = mean(y2,2);
save m2 y2 Fs2
clear filename
filename = filenames{3};
[y3,Fs3] = audioread(filename);
y3 = mean(y3,2);
save m3 y3 Fs3
clear filename
filename = filenames{4};
[y4,Fs4] = audioread(filename);
y4 = mean(y4,2);
save m4 y4 Fs4
clear filename
filename = filenames{5};
[y5,Fs5] = audioread(filename);
y5 = mean(y5,2);
save m5 y5 Fs5
clear filename
filename = filenames{6};
[y6,Fs6] = audioread(filename);
y6 = mean(y6,2);
save m6 y6 Fs6
clear filename

[trainy1, trainFs1] = audioread('TrainMoonlight.mp3');
trainy1 = mean(trainy1,2);
save trainm1 trainy1 trainFs1

[trainy4, trainFs4] = audioread('TrainVale.mp3');
trainy4 = mean(trainy4,2);
save trainm4 trainy4 trainFs4

[trainy3, trainFs3] = audioread('TrainDivine.mp3');
trainy3 = mean(trainy3,2);
save trainm3 trainy3 trainFs3

[trainy5, trainFs5] = audioread('TrainRomenJewel.mp3');
trainy5 = mean(trainy5,2);
save trainm5 trainy5 trainFs5

%% loading mat files
clear; close all; clc
load m1.mat %beethoven-moonlightsonata.mp3
load m2.mat %Deflo & Lliam Taylor - Spotlight (feat. AWA).mp3
load m3.mat %divine - joed
load m4.mat %peculate-valeoftears.mp3
load m5.mat %romanJewels.mp3
load m6.mat %sadpuppy-youhavetheblame.mp3
%p8 = audioplayer(v3,Fs3);
%playblocking(p8);

minLen = length(y1);
y2 = y2(1:minLen);
y3 = y3(1:minLen);
y4 = y4(1:minLen);
y5 = y5(1:minLen);
y6 = y6(1:minLen);

%% spectrograms
%v1 = train1'; v2 = train2'; v3 = train3'; v4 = train4'; v5 = train5'; v6 = train6'; 
v1 = y1'; v2 = y2'; v3 = y3'; v4 = y4'; v5 = y5'; v6 = y6';
n = length(v1);
L = length(v1)/Fs1;
k=(2*pi/(L))*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
t = (1:length(v1))/Fs1;
a = 125;
tslide = 0:0.1:length(v1)/Fs1;



for j = 1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2); 
    Sg1 = g.*v1; Sg2 = g.*v2; Sg3 = g.*v3; Sg4 = g.*v4; Sg5 = g.*v5; Sg6 = g.*v6;
    Sgt1 = fft(Sg1); Sgt2 = fft(Sg2); Sgt3 = fft(Sg3); Sgt4 = fft(Sg4); Sgt5 = fft(Sg5);Sgt6 = fft(Sg6);
    
    Sgt_spec1(j,:) = fftshift(abs(Sgt1));Sgt_spec2(j,:) = fftshift(abs(Sgt2));
    Sgt_spec3(j,:) = fftshift(abs(Sgt3));Sgt_spec4(j,:) = fftshift(abs(Sgt4));
    Sgt_spec5(j,:) = fftshift(abs(Sgt5));Sgt_spec6(j,:) = fftshift(abs(Sgt6));

end
%%
figure(6)
pcolor(tslide,ks,Sgt_spec1.'), 
shading interp 
set(gca,'Ylim',[0 2000],'Fontsize',16) 
colormap(hot)
hold on
figure(7)
pcolor(tslide,ks,Sgt_spec4.'), 
shading interp 
set(gca,'Ylim',[0 2000],'Fontsize',16) 
colormap(hot)

%%
feature =50;
[U,S,V,threshold,w,sortm1,sortm2] = dc_trainer(Sgt_spec1,Sgt_spec4,feature);
figure(5)
subplot(1,2,1)
histogram(sortm1,10); hold on, plot([threshold threshold],[0 2000],'r')
set(gca,'Xlim',[-100 100],'Ylim',[0 2000],'Fontsize',14)
title('Beethoven(Classical)')
subplot(1,2,2)
histogram(sortm2,10); hold on, plot([threshold threshold],[0 2000],'r')
set(gca,'Xlim',[-100 100],'Ylim',[0 2000],'Fontsize',14)
title('Vale of Tears(Heavy Metal)')
%%
load trainm1.mat %beethoven-moonlightsonata.mp3
load trainm3.mat %divine - joed
load trainm4.mat %peculate-valeoftears.mp3
load trainm5.mat %romanJewels.mp3

%p8 = audioplayer(v3,Fs3);
%playblocking(p8);
minLen = length(trainy1);
trainy3 = trainy3(1:minLen);
trainy4 = trainy4(1:minLen);
trainy5 = trainy5(1:minLen);
v1 = trainy1'; v3 = trainy3';v4 = trainy4';v5 = trainy5';
n = length(v1);
L = length(v1)/trainFs1;
k=(2*pi/(L))*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
t = (1:length(v1))/Fs1;
a = 125;
tslide = 0:0.1:length(v1)/Fs1;



for j = 1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2); 
    trainSg1 = g.*v1;  trainSg3 = g.*v3; trainSg4 = g.*v4; trainSg5 = g.*v5;
    trainSgt1 = fft(Sg1);  trainSgt3 = fft(Sg3); trainSgt4 = fft(Sg4); trainSgt5 = fft(Sg5);
    trainSgt_spec1(j,:) = fftshift(abs(trainSgt1));
    trainSgt_spec3(j,:) = fftshift(abs(trainSgt3));
    trainSgt_spec4(j,:) = fftshift(abs(trainSgt4));
    trainSgt_spec5(j,:) = fftshift(abs(trainSgt5));

end
pval = w'*trainSgt1'
%%
function [U,S,V,threshold,w,sortm1,sortm2] = dc_trainer(m1,m2,feature)
    nd = size(m1,2); nc = size(m2,2);
    
    [U,S,V] = svd([m1 m2],'econ');
    
    music = S*V'; % projection onto principal components
    U = U(:,1:feature);
    m1 = music(1:feature,1:nd);
    m2 = music(1:feature,nd+1:nd+nc);
    
    md = mean(m1,2);
    mc = mean(m2,2);
    
    Sw = 0; % within class variances
    for k=1:nd
        Sw = Sw + (m1(:,k)-md)*(m1(:,k)-md)';
    end
    for k=1:nc
        Sw = Sw + (m2(:,k)-mc)*(m2(:,k)-mc)';
    end
    
    Sb = (md-mc)*(md-mc)'; % between class 
    
    [V2,D] = eig(Sb,Sw); % linear discriminant analysis
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind); w = w/norm(w,2);
    
    vm1 = w'*m1; 
    vm2 = w'*m2;
    
    if mean(vm1)>mean(vm2)
        w = -w;
        vm1 = -vm1;
        vm2 = -vm2;
    end
    
    
    sortm1 = sort(vm1);
    sortm2 = sort(vm2);
    
    t1 = length(sortm1);
    t2 = 1;
    while sortm1(t1)>sortm2(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold = (m1(t1)+m2(t2))/2;
end