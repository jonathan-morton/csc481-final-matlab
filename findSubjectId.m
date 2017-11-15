function id = findSubjectId(recognizedIndex, train_data, faces)
id = -1;
for i = 1:size(faces, 2)
    for j = 1:size(faces(i).pictures, 2)
        if isequal(faces(i).vectors(:,j), train_data(:, recognizedIndex))
            id = faces(i).id;
            return
        end
    end
end
end