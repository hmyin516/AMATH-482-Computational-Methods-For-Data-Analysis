clear all; close all; clc;

%%
load('cam1_4.mat')
implay(vidFrames1_4)

%% Ideal Case Test 1

clear all; close all; clc;
load('cam1_1.mat')
numFrames1 = size(vidFrames1_1,4);

data1 = [];
filter = zeros (480 ,640) ;
filter (175:425 , 300:400 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames1
    X = vidFrames1_1(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 250;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data1 = [data1; mean(X), mean(Y)];
end 


load('cam2_1.mat')
numFrames2 = size(vidFrames2_1,4);
filter = zeros (480 ,640) ;
filter (100:450 , 200:350 ) = 1;
data2 = [];
for j = 1:numFrames2
    X = vidFrames2_1(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 250;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx ) ;
    data2 = [data2; mean(X), mean(Y) ];
end 


load('cam3_1.mat')
data3 = [];
filter = zeros (480 ,640) ;
filter (200:300 , 210:500 ) = 1;

numFrames3 = size(vidFrames3_1,4);
for j = 1:numFrames3
    X = vidFrames3_1(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 245;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data3 = [data3; mean(X), mean(Y) ];
end 

%% resizing ideal
[M, I] = min(data1(1:30,2));
data1 = data1(I:end,:);
[M, I] = min(data2(1:30,2));
data2 = data2(I:end,:);
[M, I] = min(data3(1:30,2));
data3 = data3(I:end,:);

data2 = data2(1:length(data1),:);
data3 = data3(1:length(data1),:);

%% PCA
dat_arr = [data1'; data2'; data3'];

[m,n]=size(dat_arr);
mn=mean(dat_arr,2);
dat_arr = dat_arr - repmat(mn,1,n);
[u,s,v]=svd(dat_arr'/sqrt(n-1));
lambda=diag(s).^2;
Y= dat_arr' * v;


%% Ideal plot



subplot(2,1,1)
plot(1:218, Y(:,6)), hold on
title("Ideal Displacement across Principal Component")
xlabel("Frames"); ylabel("Motion(pixels)")
subplot(2,1,2)
plot (1:6 , lambda/sum(lambda) ,'o', 'Linewidth', 2);
title("Energy of Diagonal Variacne");
xlabel("Diagonal Variance"); 
ylabel("Energy")
%% Noisy Case
clear all; close all; clc;
load('cam1_2.mat')
numFrames1 = size(vidFrames1_2,4);

data1 = [];
filter = zeros (480 ,640) ;
filter (175:425 , 300:450 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames1
    X = vidFrames1_2(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 250;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data1 = [data1; mean(X), mean(Y)];
end 


load('cam2_2.mat')
numFrames2 = size(vidFrames2_2,4);

data2 = [];
filter = zeros (480 ,640) ;
filter (50:450 , 150:450 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames2
    X = vidFrames2_2(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 245;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data2 = [data2; mean(X), mean(Y)];
end 


load('cam3_2.mat')
numFrames3 = size(vidFrames3_2,4);

data3 = [];
filter = zeros (480 ,640) ;
filter (200:400 , 210:500 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames1
    X = vidFrames3_2(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 245;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data3 = [data3; mean(X), mean(Y)];
end
%% resizing noisy
[M, I] = min(data1(1:30,2));
data1 = data1(I:end,:);
[M, I] = min(data2(1:30,2));
data2 = data2(I:end,:);
[M, I] = min(data3(1:30,2));
data3 = data3(I:end,:);

data2 = data2(1:length(data1),:);
data3 = data3(1:length(data1),:);
%% PCA Noisy
dat_arr = [data1'; data2'; data3'];
[m,n] = size(dat_arr);

mn=mean(dat_arr,2);
dat_arr = dat_arr - repmat(mn,1,n);
[u,s,v]=svd(dat_arr'/sqrt(n-1));
lambda=diag(s).^2;
Y=dat_arr' * v;

%% noisy plots
figure()


subplot(2,1,1)
plot(1:287, Y(:,2))
title("Noisy Displacement across Principal Component")
xlabel("Frames"); ylabel("Motion(pixels)")
subplot(2,1,2)
plot (1:6 , lambda/sum(lambda) ,'o', 'Linewidth', 2);
title("Energy of Diagonal Variacne");
xlabel("Diagonal Variance"); 
ylabel("Energy")
%% Horizontal Displacment
clear all; close all; clc;
load('cam1_3.mat')
numFrames1 = size(vidFrames1_3,4);

data1 = [];
filter = zeros (480 ,640) ;
filter (200:450 , 300:450 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames1
    X = vidFrames1_3(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 250;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data1 = [data1; mean(X), mean(Y)];
end 

load('cam2_3.mat')
numFrames2 = size(vidFrames2_3,4);

data2 = [];
filter = zeros (480 ,640) ;
filter (100:450 , 150:425 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames2
    X = vidFrames2_3(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 245;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data2 = [data2; mean(X), mean(Y)];
end 


load('cam3_3.mat')
numFrames3 = size(vidFrames3_3,4);

data3 = [];
filter = zeros (480 ,640) ;
filter (150:360 , 210:500 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames3
    X = vidFrames3_3(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 245;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data3 = [data3; mean(X), mean(Y)];
end
%% resizing horiz dis
[M, I] = min(data1(1:30,2));
data1 = data1(I:end,:);
[M, I] = min(data2(1:30,2));
data2 = data2(I:end,:);
[M, I] = min(data3(1:30,2));
data3 = data3(I:end,:);

data2 = data2(1:length(data1),:);
data3 = data3(1:length(data1),:);

%% PCA  horiz
dat_arr = [data1'; data2'; data3'];
[m,n] = size(dat_arr);

mn=mean(dat_arr,2);
dat_arr = dat_arr - repmat(mn,1,n);
[u,s,v]=svd(dat_arr'/sqrt(n-1));
lambda=diag(s).^2;
Y=dat_arr' * v;

%% hoirz plots
figure()


subplot(2,1,1)
plot(1:210, Y(:,1),1:210, Y(:,2),1:210, Y(:,3))
title("Horizontal Displacement across Principal Component")
xlabel("Frames"); ylabel("Motion(pixels)")
subplot(2,1,2)
plot (1:6 , lambda/sum(lambda) ,'o', 'Linewidth', 2);
title("Energy of Diagonal Variacne");
xlabel("Diagonal Variance"); 
ylabel("Energy")

%% Rotational and Horizotnal

clear all; close all; clc;
load('cam1_4.mat')
numFrames1 = size(vidFrames1_4,4);

data1 = [];
filter = zeros (480 ,640) ;
filter (230:450 , 280:450 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames1
    X = vidFrames1_4(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 245;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data1 = [data1; mean(X), mean(Y)];
end 

load('cam2_4.mat')
numFrames2 = size(vidFrames2_4,4);

data2 = [];
filter = zeros (480 ,640) ;
filter (100:450 , 150:400 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames2
    X = vidFrames2_4(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 245;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data2 = [data2; mean(X), mean(Y)];
end 


load('cam3_4.mat')
numFrames3 = size(vidFrames3_4,4);

data3 = [];
filter = zeros (480 ,640) ;
filter (100:300 , 260:600 ) = 1;

%x 480, y 640
% x and y are swapped
for j = 1:numFrames3
    X = vidFrames3_4(:,:,:,j);
    G = double(rgb2gray(X));
    Xf = G .* filter ;
    th = Xf > 230;
    indx = find(th) ;
    [Y ,X] = ind2sub (size(th) , indx) ;
    data3 = [data3; mean(X), mean(Y)];
end
%% resizing r/horiz dis
[M, I] = min(data1(1:10,2));
data1 = data1(I:end,:);
[M, I] = min(data2(1:10,2));
data2 = data2(I:end,:);
[M, I] = min(data3(1:10,2));
data3 = data3(I:end,:);

data2 = data2(1:length(data1),:);
data3 = data3(1:length(data1),:);

%% PCA  horiz
dat_arr = [data1'; data2'; data3'];
[m,n] = size(dat_arr);

mn=mean(dat_arr,2);
dat_arr = dat_arr - repmat(mn,1,n);
[u,s,v]=svd(dat_arr'/sqrt(n-1));
lambda=diag(s).^2;
Y=dat_arr' * v;
var = diag(s);
%% hoirz plots
figure()


subplot(2,1,1)
plot(1:384, Y(:,1),1:384, Y(:,2),1:384 , Y(:,3));
title("Horizontal Displacement across Principal Component")
xlabel("Frames"); ylabel("Motion(pixels)")
subplot(2,1,2)
plot (1:6 , lambda/sum(lambda) ,'o', 'Linewidth', 2);
title("Energy of Diagonal Variacne");
xlabel("Diagonal Variance"); 
ylabel("Energy")