function id = getNumberId(text_id)
    id = text_id;
    id = id(2:end);
    id = str2num(id);
end