clear all, close all, clc;

num_imgs = 6500;
num_sp = 256;
dim_feat = 1024;
feat_name = 'feat_16_sample';

%seg_root = '~/hdd/ILSVRC2012/resize_256x256/val/seg/';
seg_root = '~/hdd/ILSVRC2012/resize_256x256/train/seg/';
listFile = '~/research/color/data/filelist/train_list/train_sample_rand.txt';
fid = fopen(listFile, 'r');
imgNames = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);

seg_mat = zeros(num_sp, num_imgs);
num_fg = 0;
%for i = 1 : num_imgs
for i=1:length(imgNames{1})
    i
    tic;
    %seg_name = strcat('ILSVRC2012_val_0000', sprintf('%04d', i), '.dat');
    %fid = fopen(strcat(seg_root, seg_name), 'r');
    %segs = fread(fid, num_sp, 'uint8');
    fid = fopen(strcat(seg_root, imgNames{1}{i}(1:end-4), 'dat'), 'r');
    segs = fread(fid, num_sp, 'int');
    seg_mat(:, i) = segs;
    fclose(fid);
    num_fg = num_fg + sum(segs);
    toc;
end

%fid = fopen('~/research/bh_tsne/data/seg_mat.dat', 'w');
%fwrite(fid, seg_mat);
%fclose(fid);

fid = fopen('~/research/bh_tsne/data/seg_sample_mat.dat', 'w');
fwrite(fid, seg_mat, 'int');
fclose(fid);

% num_fg = ;
tic;
fid = fopen(strcat('~/research/ResNet18/data/', feat_name, '.dat'), 'r');
X = fread(fid, num_sp * num_imgs * dim_feat, 'float32');
fclose(fid);
X = reshape(X, [dim_feat, num_sp * num_imgs]);
X = X';
toc;
Y = zeros(num_fg, dim_feat);
%fid = fopen('~/research/bh_tsne/data/seg_mat.dat', 'r');
%seg_mat = fread(fid, num_sp*num_imgs, 'uint8');
fid = fopen('~/research/bh_tsne/data/seg_sample_mat.dat', 'r');
seg_mat = fread(fid, num_sp*num_imgs, 'int');
fclose(fid);
seg_mat = reshape(seg_mat, [num_sp, num_imgs]);
cnt = 1;
for i = 1 : num_imgs
    tic;
    %seg_name = strcat('ILSVRC2012_val_0000', sprintf('%04d', i), '.dat');
    %fid = fopen(strcat(seg_root, seg_name), 'r');
    %segs = fread(fid, num_sp, 'uint8');
    %fclose(fid);
    segs = seg_mat(:, i);
    for j = 1 : num_sp
        if segs(j) == 1
            Y(cnt, :) = X(num_sp * (i - 1) + j, :);
            cnt = cnt + 1;
        end
    end
    toc;
end

map = fast_tsne(Y);
fid = fopen(strcat('~/research/bh_tsne/data/', feat_name, '_tsne.dat'), 'w');
fwrite(fid, map', 'float32');
fclose(fid);
