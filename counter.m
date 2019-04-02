function [count] = counter(data)
    
    [rows, clmns] = size(data);
    count=zeros(10,1);
    
    for i = 1:rows
        count(data(i)) = count(data(i)) + 1;
    end
    
end