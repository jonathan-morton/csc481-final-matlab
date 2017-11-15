% Jonathan Morton and Jun Son

% Run this script to train your data
% Clear memory and console
close all
clear
clc

rootFolder = 'att_faces';
image_dims = [112, 92];

imageFolders = dir(rootFolder);
imageFolders = {imageFolders.name};
imageFolders = imageFolders(startsWith(imageFolders(:), 's'));

images = [];
subjectIds = {};

for i = 1:size(imageFolders, 2) 
    folder = imageFolders{i};
    subjectPath = strcat(rootFolder, '/',folder);
    images = dir(subjectPath);
    images = {images.name};
    images = images(or(endsWith(images(:), 'pgm'), endsWith(images(:), 'png')));
    
    subjectId = folder;
    subjectIds = unique([subjectIds subjectId]);
    faces(i).id = subjectId;
    idNumber = subjectId(2:end);
    faces(i).idNumber = str2num(idNumber);
    faces(i).pictures = {};
    faces(i).vectors = [];
    
    
    for j = 1:size(images, 2)
        fullImagePath = strcat(subjectPath, '/', images{j});
        image = imread(fullImagePath);
        image = double(image);
        
        imageIndex = images{j};
        imageIndex = imageIndex(1:end-4);
        imageIndex = str2num(imageIndex);
        faces(i).pictures{imageIndex} = image;
        
        % read and convert images to column vector
        currentPicture = faces(i).pictures{imageIndex};
        [irow icol] = size(currentPicture);
        faceVector = reshape(currentPicture',irow*icol,1);
        faces(i).vectors = [faces(i).vectors faceVector];
    end
    
    % Reconstructed average face
%     avgFace = reshape(mean_face, 112, 92);
%     figure; imshow(avgFace);
end

faceFields = fieldnames(faces);
faceCell = struct2cell(faces);
faceSize = size(faceCell);  
% Convert to a matrix
faceCell = reshape(faceCell, faceSize(1), []);      % Px(MxN)
% Make each field a column
faceCell = faceCell';                         % (MxN)xP
% Sort by first field "idNumber"
faceCell = sortrows(faceCell, 2);
faceCell = reshape(faceCell', faceSize);
% Convert to Struct
faces = cell2struct(faceCell, faceFields, 1);

% split into training and testing data with 1:9 split ratio
% our faces are in test_data(:,n) where n = [71,74]
train_data = [];
test_data = [];
test_data_id = [];
train_data_id = [];

test_index = [1];
for i = 1:size(faces,2)
    for j = 1:size(faces(i).vectors, 2)
        faceVector = faces(i).vectors(:,j);
        if ismember(j, test_index)
            test_data = [test_data, faceVector];
            test_data_id = [test_data_id getNumberId(faces(i).id)];
        else
            train_data = [train_data, faceVector];
            train_data_id = [train_data_id getNumberId(faces(i).id)];
        end
    end
end

mean_face = mean(train_data, 2);

avgFace = reshape(mean_face, 92, 112);
figure; imshow(uint8(avgFace'));

Class_number = (size(faces, 2)); % Number of classes (or persons)
Class_population = size(faces(1).vectors,2) - size(test_index,2); % Number of images in each class
P = Class_population * Class_number; % Total number of training images

shifted_images = train_data - repmat(mean_face, 1, P);

run('face_recognition_training.m')
run('face_recognition_test.m')
run('face_recognition_evaluation.m')
