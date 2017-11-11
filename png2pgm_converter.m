
% I used this script to convert your png-format images to pgm-format. 

for i=1:10
    s = sprintf('%i.png',i);
    r = sprintf('%i.pgm',i);
    X=imread(s);
    imwrite(X,r)
end