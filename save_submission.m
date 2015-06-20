function save_submission(P, filename)
% SAVE_SUBMISSION Save a submission suitable for kaggle.
%     SAVE_SUBMISSION(P, [FILENAME])
%
% Inputs:
%   P - n-by-9 matrix of probabilities
%   filename - OPTIONAL name of the file to create.

if nargin < 2
    filename = 'submission.csv';
end

% Write header
fileID = fopen(filename,'w');
fprintf(fileID, 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n');
fclose(fileID);

% Write data
ids = (1:size(P, 1))';
dlmwrite(filename, [ids P], '-append', 'precision', 10);

end % function
