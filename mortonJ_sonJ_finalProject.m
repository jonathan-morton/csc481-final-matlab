% Jonathan Morton and Jun Son
rootFolder = 'att_faces';
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
    faces(i).pictures = {};
    faces(i).vectors = [];
    
    for j = 1:size(images, 2)
        fullImagePath = strcat(subjectPath, '/', images{j});
        image = imread(fullImagePath);
        faces(i).pictures = [faces(i).pictures, image];
        
        % read and convert images to column vector
        currentPicture = faces(i).pictures{j};
        faceVector = reshape(currentPicture,size(currentPicture,1)*size(currentPicture,2),1);
        faces(i).vectors = [faces(i).vectors, faceVector];
    end
    
    mean_face = uint8(round(mean(faces(i).vectors, 2)));
    shiftedImages = faces(i).vectors - repmat(mean_face, 1, size(faces(i).vectors, 2));
    
    % Reconstructed average face
    % avgFace = reshape(mean_face, 112, 92)
end