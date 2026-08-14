info = dicominfo('.dcm');
img  = dicomread(info);
img  = double(img);   % work in double precision for now

bitsStored = info.BitsStored;   % CHAOS MRI is typically 12-bit
maxVal = 2^bitsStored - 1;