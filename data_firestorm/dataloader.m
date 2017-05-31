function S = dataloader(filename)

    % Load Headers
    delimiter = ',';
    endRow = 1;

    formatSpec = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%[^\n\r]';

    fileID = fopen(filename,'r');

    headerArray = textscan(fileID, formatSpec, endRow, 'Delimiter', delimiter, 'ReturnOnError', false);

    % Load Data

    delimiter = ',';
    startRow = 2;
    formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);

    fclose(fileID);

    S = struct();
    for i = 1:length(headerArray)
        S.(char(headerArray{i})) = dataArray{:,i};
    end

end