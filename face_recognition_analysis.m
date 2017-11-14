

% display mean_face for train_data
% example input image = s1-2.pgm
input = colVec2image(train_data(:,1));
mean = colVec2image(mean_face);
shifted = colVec2image(shifted_images(:,1));
figure,subplot(1,3,1),imagesc(input), colormap(gray), title('Input image');
subplot(1,3,2),imagesc(mean), colormap(gray), title('Mean face');
subplot(1,3,3),imagesc(shifted), colormap(gray), title('Mean-shifted face');
[rowDim, colDim]=size(input);





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








