
% Run this script to train your data
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

%%%%%%%%%%%%%%%%%%%%%%%% Snapshot method of Eigenface algorithm
A = shifted_images;
L = A'*A; % L is the surrogate of covariance matrix C=A*A'. Original: L = cov(A');
[V D] = eig(L); % Diagonal elements of D are the eigenvalues for both L=A'*A and C=A*A'.

%%%%%%%%%%%%%%%%%%%%%%%% sort eigenvalues in descending order
eigval = diag(D);
eigval = eigval(end:-1:1);

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
