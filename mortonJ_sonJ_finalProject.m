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
    
    for j = 1:size(images, 2)
        fullImagePath = strcat(subjectPath, '/', images{j});
        image = imread(fullImagePath);
        faces(i).pictures = [faces(i).pictures, image];
    end
end



