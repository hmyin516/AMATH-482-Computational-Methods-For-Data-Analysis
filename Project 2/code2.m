
%% loading handel's messiah
clear; close all; clc
load handel
v = y';

plot((1:length(v))/Fs,v);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');

%% play muisc
p8 = audioplayer(v,Fs);
playblocking(p8);


%% Gabor Spectograms
L=9; n=length(v);
t2=linspace(0,L,n+1); t=t2(1:n); 

k=(2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1]; 
ks=fftshift(k);


a_vec = [1000 100 10 1];

for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.09:9;
    Sgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        Sg=g.*v; 
        Sgt=fft(Sg); 
        Sgt_spec(j,:) = fftshift(abs(Sgt)); 
    end

    subplot(2,2,jj)
    pcolor(tslide,ks,Sgt_spec.'), shading interp 
    title(['Gaussian Filter a = ',num2str(a)],'Fontsize',16)
    
    xlabel('Time (t)'), ylabel('frequency)')
    colormap(hot) 
end


%% Filter plots for oversampling/undersampling


a = 500;
tslide=0:0.1:9;

for j=1:50
    g=exp(-a*(t-tslide(j)).^2);  
    Sg=g.*v; 
    Sgt=fft(Sg); 
    
    subplot(3,1,1) 
    plot(t,v,'k','Linewidth',2) 
    hold on 
    plot(t,g,'m','Linewidth',2)
    hold off
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('S(t)')

    subplot(3,1,2) 
    plot(t,Sg,'k','Linewidth',2) 
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('Sg(t)')

    subplot(3,1,3) 
    plot(ks,abs(fftshift(Sgt))/max(abs(Sgt)),'r','Linewidth',2);
    set(gca,'Fontsize',16)
    xlabel('frequency (\omega)'), ylabel('FFT(Sg)')
    drawnow
    pause(0.1)
end

%% Calculate Gabor transform and plot spectrogram
a = 1;
tslide=0:0.1:10;
Sgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    Sg=g.*S; 
    Sgt=fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(6)
pcolor(tslide,ks,Sgt_spec.'), 
shading interp 
set(gca,'Ylim',[-50 50],'Fontsize',16) 
colormap(hot)
%% mexican hat wavelet
L=9; n=length(y);
t2=linspace(0,L,n+1); t=t2(1:n); 
k=(2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1]; 
ks=fftshift(k);
St = fft(v);

a_vec = [1000 100 10 1];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:1:9;
    Sgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g = (1 - (t - tslide(j)).^2).*exp(-a*((t-tslide(j)).^2)/2);
        Sg=g.*v; 
        Sgt=fft(Sg); 
        Sgt_spec(j,:) = fftshift(abs(Sgt)); 
    end

    subplot(2,2,jj)
    pcolor(tslide,ks,Sgt_spec.'), 
    shading interp 
    title(['Mexican Hat Filter a = ',num2str(a)],'Fontsize',16)
     
    xlabel('Time (t)'), ylabel('frequency')
    colormap(hot) 
end


%% Part 2, Piano
clear; close all; clc;
[y,Fs] = audioread('music1.wav');
tr_piano=length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');
%p8 = audioplayer(y,Fs); playblocking(p8);
v = y';

%% spectrogram

n = length(v);
L = length(v)/Fs;
k=(2*pi/(L))*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
t = (1:length(v))/Fs;

a = 500; 
tslide = 0:.2:16; 
Sgt_spec = zeros(length(tslide),n);

figure(1)
for j = 1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2); 


    Sg = g.*v; 
    Sgt = fft(Sg);
    
    
    Sgt_spec(j,:) = fftshift(abs(Sgt));
    
    subplot(3,1,1)
    plot(t,v,'k',t,g,'r');
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('S(t)')
    axis([0 L -0.5 1]);
    subplot(3,1,2)
    plot(t, Sg);
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('Sg(t)')
    axis([0 L -0.5 1]);
    subplot(3,1,3)
    plot(ks,abs(fftshift(Sgt))/max(abs(Sgt)),'r','Linewidth',2);
    xlabel('frequency (\omega)'), ylabel('FFT(Sg)')
    pause(0.1)
end

figure(2)
subplot(1,1,1)
pcolor(tslide,ks/(2*pi),Sgt_spec.'), shading interp
axis([0 15 200 400])
set(gca,'Fontsize',16)
xlabel('time(s)'), ylabel('frequency(Hz)')

colormap(hot)
hold on
title('Mary had a little lamb (piano)');
hold on;


pb = plot([0 16],[246.94 246.94],'c') %B_3 
hold on;
pc = plot([0 16],[261.63 261.63],'b'); %C_4 
hold on;

pd = plot([0 16],[293.66 293.66],'y'); %D_4 
hold on;

pe = plot([0 16],[329.63 329.63],'Color',[.61 .51 .74]) %E_4 
hold on;

leg = legend([pb,pc,pd,pe],{'B3','C4','D4','E4',},'Orientation','horizontal')
leg.Title.String = 'Frequencies of Notes'
%% Part 2 Recorder
clear; close all; clc;
[y,Fs] = audioread('music2.wav');
tr_recorder=length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');
%p8 = audioplayer(y,Fs); playblocking(p8);
v = y';
%% spectrogram
n = length(v);
L = length(v)/Fs;
k=(2*pi/(L))*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
t = (1:length(v))/Fs;

a = 500; 
tslide = 0:1:16; 
Sgt_spec = zeros(length(tslide),n);

figure(1)
for j = 1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2); 
    Sg = g.*v; 
    Sgt = fft(Sg);
    
    
    Sgt_spec(j,:) = fftshift(abs(Sgt));
    
    subplot(3,1,1)
    plot(t,v,'k',t,g,'r');
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('S(t)')
    axis([0 L -0.5 1]);
    subplot(3,1,2)
    plot(t, Sg);
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('Sg(t)')
    axis([0 L -0.5 1]);
    subplot(3,1,3)
    plot(ks,abs(fftshift(Sgt))/max(abs(Sgt)),'r','Linewidth',2);
    xlabel('frequency (\omega)'), ylabel('FFT(Sg)')
    pause(0.1)
end
figure(2)
subplot(1,1,1)
pcolor(tslide,ks/(2*pi),Sgt_spec.'), shading interp
axis([0 15 0 1500])
set(gca,'Fontsize',16)
xlabel('time(s)'), ylabel('frequency(Hz)')
title('Mary had a little lamb (recorder)');
hold on;
pg = plot([0 16],[783.99 783.99],'c') %G4 freq
hold on;

pa = plot([0 16],[880.00 880.00],'g'); %A4 freq
hold on;

pb=plot([0 16],[987.77 987.77],'r'); %B4 freq
hold on;
pc =plot([0 16],[1046.50 1046.50],'w') %C5 freq
colormap(hot)
hold on
leg = legend([pg,pa,pb,pc],{'G4','A4','B4','C4',},'Orientation','horizontal')
leg.Title.String = 'Frequencies of Notes'
