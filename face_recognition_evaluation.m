
% Run this code to evaluate your data

recognition_count = 0 ;

for i = 1:Class_number
    test_image = test_data(:,i);
    Difference = test_image - mean_face; % Centered test image
    ProjectedTestImage = V_Fisher' * V_PCA' * Difference; % Test image feature vector
    
    Train_Number = size(ProjectedImages_Fisher,2);
    Euc_dist = [];
    for j = 1 : Train_Number
        q = ProjectedImages_Fisher(:,j);
        temp = ( norm( ProjectedTestImage - q ) )^2;
        Euc_dist = [Euc_dist temp];
    end
    
    [Euc_dist_min , Recognized_index] = min(Euc_dist);
    
    if train_data_id(Recognized_index) == i
%         fprintf('Matched!\n');
        recognition_count = recognition_count+ 1;
    else
        fprintf('NO MATCH! TestId: %i TrainingId: %i\n',i, train_data_id(Recognized_index));
    end

%     classId = findSubjectId(Recognized_index,train_data,faces);
%     classNumberId = getIdFromIndex(Class_number, faces);
%     if strcmp(classId, classNumberId)
%         recognition_count = recognition_count + 1;
%     end
    
end

recognition_rate = recognition_count/Class_number ;
fprintf('Recognition rate for test images: %g\n', recognition_rate); 
