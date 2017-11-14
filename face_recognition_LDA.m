
% Clear memory and console
close all
clear
clc

% get directories for all images
% please use 'getfn.m' with this script
%'C:\Users\json13\Desktop\att_faces'
% 'E:\att_faces'
fn = getfn('E:\ORL_images', 'pgm$')

% read and convert images to column vector
image_dims = [112, 92];
num_images = numel(fn);
images = [];
for n = 1:num_images
    img = imread(fn{n});
    [irow icol] = size(img);
    temp = reshape(img',irow*icol,1);   % Reshaping 2D images into 1D image vectors
    images = [images temp];
end
images = double(images);

% split into training and testing data with 2:8 split ratio
% our faces are in test_data(:,n) where n = [71,74]
train_data = [];
test_data = [];
cont = 1;
for i=1:42
    for j=1:10
        if j == 1
            test_data = [test_data,images(:,cont)];
        else
            train_data = [train_data,images(:,cont)];
        end
        cont = cont + 1;
    end 
end

Class_number = ( size(train_data,2) )/9; % Number of classes (or persons)
Class_population = 9; % Number of images in each class
P = Class_population * Class_number; % Total number of training images

%%%%%%%%%%%%%%%%%%%%%%%% calculating the mean image 
mean_face = mean(train_data, 2);

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the deviation of each image from mean image
% shifted_images = bsxfun(@minus, train_data, mean_face);
shifted_images = train_data - repmat(mean_face, 1, P);


% display mean_face for train_data
% example input image = s1-3.pgm
input = colVec2image(train_data(:,1));
mean = colVec2image(mean_face);
shifted = colVec2image(shifted_images(:,1));
figure,subplot(1,3,1),imagesc(input), colormap(gray), title('Input image');
subplot(1,3,2),imagesc(mean), colormap(gray), title('Mean face');
subplot(1,3,3),imagesc(shifted), colormap(gray), title('Mean-shifted face');
[rowDim, colDim]=size(input);





%%%%%%%%%%%%%%%%%%%%%%%% Snapshot method of Eigenface algorithm
A = shifted_images;
L = A'*A; % L is the surrogate of covariance matrix C=A*A'. Original: L = cov(A');
[V D] = eig(L); % Diagonal elements of D are the eigenvalues for both L=A'*A and C=A*A'.

%%%%%%%%%%%%%%%%%%%%%%%% sort eigenvalues in descending order
eigval = diag(D);
eigval = eigval(end:-1:1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% you may skip this part 
%%%%%%%%%%%%%%%%%%%%%%%% evaluate the number of principal components needed to represent 95% Total variance.
eigsum = sum(eigval);
csum = 0;
for i = 1:10304
    csum = csum + eigval(i);
    tv = csum/eigsum;
    if tv > 0.95
        k95 = i;
        break
    end ;
end;
fprintf('The number of principal components to represent 95 percent total variance: %g\n', i); 

% ====== Plot variance percentage vs. no. of eigenvalues
cumVar=cumsum(eigval);
cumVarPercent=cumVar/cumVar(end)*100;
plot(cumVarPercent, '.-');
xlabel('No. of eigenvalues');
ylabel('Cumulated variance percentage (%)');
title('Variance percentage vs. no. of eigenvalues');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%% 
L_eig_vec = [];
for i = P:-1:Class_number+1
    L_eig_vec = [L_eig_vec V(:,i)];
end

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the eigenvectors of covariance matrix 'C'
V_PCA = A * L_eig_vec; % A: centered image vectors

%%%%%%%%%%%%%%%%%%%%%%%% Projecting centered image vectors onto eigenspace
ProjectedImages_PCA = V_PCA'*A;





%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%%        LDA
%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Calculating the mean of each class in eigenspace
m_PCA = mean(ProjectedImages_PCA,2); % Total mean in eigenspace
m = zeros(P-Class_number,P-Class_number); 
Sw = zeros(P-Class_number,P-Class_number); % Initialization os Within Scatter Matrix
Sb = zeros(P-Class_number,P-Class_number); % Initialization of Between Scatter Matrix

for i = 1 : Class_number
    m = mean(ProjectedImages_PCA(:,Class_population*i-(Class_population-1):Class_population*i), 2 )';    
     
    for j = ( (i-1)*Class_population+1 ) : ( i*Class_population )
        Sw = Sw + (ProjectedImages_PCA(:,j)-m)*(ProjectedImages_PCA(:,j)-m)'; % Within Scatter Matrix
    end
    
    Sb = Sb + (m-m_PCA) * (m-m_PCA)'; % Between Scatter Matrix
end
Sb = Class_population*Sb;


%%%%%%%%%%%%%%%%%%%%%%%% Calculating Fisher discriminant basis's
% We want to maximise the Between Scatter Matrix, while minimising the
% Within Scatter Matrix. Thus, a cost function J is defined, so that this condition is satisfied.
J = inv(Sw) * Sb;
[J_eig_vec, J_eig_val] = eig(J); 
V_Fisher = fliplr(J_eig_vec);



%%%%%%%%%%%%%%%%%%%%%%%% Projecting images onto Fisher linear space
% Yi = V_Fisher' * V_PCA' * (Ti - m_database) 
ProjectedImages_Fisher = V_Fisher' * ProjectedImages_PCA;


a = V_Fisher' * V_PCA';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Extracting the FLD features from test image
InputImage = imread('s10.pgm');
temp = InputImage(:,:,1);

[irow icol] = size(temp);
InImage = reshape(temp',irow*icol,1);

Difference = double(InImage) - mean_face; % Centered test image
ProjectedTestImage = V_Fisher' * V_PCA' * Difference; % Test image feature vector

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Euclidean distances 
% Euclidean distances between the projected test image and the projection
% of all centered training images are calculated. Test image is
% supposed to have minimum distance with its corresponding image in the
% training database.
Train_Number = size(ProjectedImages_Fisher,2);
Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages_Fisher(:,i);
    temp = ( norm( ProjectedTestImage - q ) )^2;
    Euc_dist = [Euc_dist temp];
end

[Euc_dist_min , Recognized_index] = min(Euc_dist);


%%%%%%%%%%%%%%%%%%%%%%%% Recognition test result
recognized = train_data(:,Recognized_index);
recognized = colVec2image(recognized);
figure, subplot(1,2,1), imagesc(InputImage), colormap(gray), title('input image');
subplot(1,2,2), imagesc(recognized), colormap(gray), title('recognized image');


















%%%%%%%%%%%%%%%%%%%%%%%% Display the first few eigenfaces (pca)
reducedDim=16;			
eigenfaces = reshape(V_PCA, icol, irow, size(V_PCA,2));
side=ceil(sqrt(reducedDim));
for i=1:reducedDim
	subplot(side,side,i);
    I = imrotate(eigenfaces(:,:,i),270);
	imagesc(I); axis image; colormap(gray);
	set(gca, 'xticklabel', ''); set(gca, 'yticklabel', '');
end


%%%%%%%%%%%%%%%%%%%%%%%% difference between the original and projected
%%%%%%%%%%%%%%%%%%%%%%%% image (pca)
origFace=train_data(:,1);
projFace=V_PCA*(V_PCA'*(origFace-mean_face))+mean_face;

I = imrotate(reshape(origFace, icol, irow),270);
I2 = imrotate(reshape(projFace, icol, irow),270);
I3 = imrotate(reshape(origFace-projFace, icol, irow),270);
subplot(1,3,1);
imagesc(I); axis image; colormap(gray); title('Original image');
subplot(1,3,2);
imagesc(I2); axis image; colormap(gray); title('Projected image');
subplot(1,3,3);
imagesc(I3); axis image; colormap(gray); title('Difference');
fprintf('Difference between orig. and projected images = %g\n', norm(origFace-projFace)); 








