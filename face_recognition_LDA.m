
% Clear memory and console
close all
clear
clc

% get directories for all images
% please use 'getfn.m' with this script
fn = getfn('C:\Users\json13\Desktop\att_faces', 'pgm$')

% read and convert images to column vector
image_dims = [112, 92];
num_images = numel(fn);
images = [];
for n = 1:num_images
    img = imread(fn{n});
    if n == 1
        images = zeros(prod(image_dims), num_images);
    end
    images(:, n) = img(:); % convert column vector
end

% split into training and testing data with 2:8 split ratio
% our faces are in test_data(:,n) where n = [71,74]
train_data = [];
test_data = [];
cont = 1;
for i=1:42
    for j=1:10
        if j == 1 | j == 3
            test_data = [test_data,images(:,cont)];
        else
            train_data = [train_data,images(:,cont)];
        end
        cont = cont + 1;
    end 
end
% size of training and testing data set
[num_vals, num_train] = size(train_data)
[num_vals, num_test] = size(test_data)

% steps 1 and 2: find the mean image and the mean-shifted input images
mean_face = mean(train_data, 2);
shifted_images = train_data - repmat(mean_face, 1, num_train);
% display mean_face for train_data
% example input image = s1-3.pgm
input = colVec2image(train_data(:,1));
mean = colVec2image(mean_face);
shifted = colVec2image(shifted_images(:,1));
figure,subplot(1,3,1),imshow(input), title('Input image');
subplot(1,3,2),imshow(mean), title('Mean face');
subplot(1,3,3),imshow(shifted), title('Mean-shifted face');
[rowDim, colDim]=size(input);



% reference: http://mirlab.org/jang/books/dcpr/fePca4fr.asp?title=11-4%20PCA%20for%20Face%20Recognition
% ====== Perform PCA
% A2 = the principal component coefficients
[A2, eigVec, eigValue]=pca(train_data);

% ====== Plot variance percentage vs. no. of eigenvalues
cumVar=cumsum(eigValue);
cumVarPercent=cumVar/cumVar(end)*100;
plot(cumVarPercent, '.-');
xlabel('No. of eigenvalues');
ylabel('Cumulated variance percentage (%)');
title('Variance percentage vs. no. of eigenvalues');

% ====== Display the first few eigenfaces
reducedDim=16;			
eigenfaces = reshape(eigVec, rowDim, colDim, size(A2,2));
side=ceil(sqrt(reducedDim));
for i=1:reducedDim
	subplot(side,side,i);
	imagesc(eigenfaces(:,:,i)); axis image; colormap(gray);
	set(gca, 'xticklabel', ''); set(gca, 'yticklabel', '');
end



% ====== difference between the original and projected image
eigVec2=eigVec(:, 1:28);			% Take the first 28 eigenvectors
origFace=train_data(:,74)
meanFace=mean_face;
projFace=eigVec2*(eigVec2'*(origFace-meanFace))+meanFace;

subplot(1,3,1);
imagesc(reshape(origFace, rowDim, colDim));
axis image; colormap(gray); title('Original image');
subplot(1,3,2);
imagesc(reshape(projFace, rowDim, colDim));
axis image; colormap(gray); title('Projected image');
subplot(1,3,3);
imagesc(reshape(origFace-projFace, rowDim, colDim));
axis image; colormap(gray); title('Difference');
fprintf('Difference between orig. and projected images = %g\n', norm(origFace-projFace)); 





