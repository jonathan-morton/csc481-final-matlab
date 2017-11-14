
function res = colVec2image(col_vec)
% a function to convert column vector to original image format
% input = column vector
% image_dims = [112, 92];

img = reshape(col_vec,92,112);
res = uint8(img');