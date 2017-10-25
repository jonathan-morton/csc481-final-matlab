% Jonathan Morton and Jun Son
imageFiles = dir('yalefaces');
images = [];
subjectIds = [];
for i = 1:size(imageFiles, 1) 
    fileName = imageFiles(i).name;
    startSplit = 'subject';
    endSplit = '.';
    
    if ~contains(fileName, startSplit, 'IgnoreCase', true)
        continue
    end
    
    subjectId = extractBetween(fileName, startSplit, endSplit);
    subjectId = str2double(subjectId{1});
    
    if ismember(subjectId, subjectIds)
         faces(subjectId).pictures{end+1} = fileName;
    else
        subjectIds = unique([subjectIds subjectId]);
        faces(subjectId).id = subjectId;
        faces(subjectId).pictures = {fileName};
    end
end
    
