
function res = colVec2image(col_vec)
% a function to convert column vector to original image format
% input = column vector
% e.g. input = test_data(:,72);
%      res = colVec2image(input);

image_dims = [112, 92]; % you may change the dimension if you need
img = reshape(col_vec,image_dims);
res = uint8(img);