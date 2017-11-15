
% Run this code to test your image

%%%%%%%%%%%%%%%%%%%%%%%% Extracting the FLD features from test image
InputImage = faces(1).pictures{1};
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


