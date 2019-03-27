function [dataoutnet] = parseClassify(typ_ochorenia)

    [rows,columns]=size(typ_ochorenia);
    
    for i=1:rows
        dataoutnet(:,i)=zeros(10,1);
        dataoutnet(typ_ochorenia(i),i)=1;
    end
    
end

