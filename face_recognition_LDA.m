
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
    images(:, n) = img(:);
end

% split into training and testing data with 2:8 split ratio
train_data = [];
test_data = [];
cont = 1;
for i=1:42
    for j=1:10
        if j>2
            train_data = [train_data,images(:,cont)];
        else
            test_data = [test_data,images(:,cont)];
        end
        cont = cont + 1;
    end 
end

