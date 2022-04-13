function [PSNR, RSE, SSIM] = ImageQualityMeasure(Xtrue, Xhat)
xSize = size(Xtrue);
PSNR = PSNR_RGB(Xhat, Xtrue);
RSE  = perfscore(Xhat, Xtrue);

SSIMArr = zeros(1,xSize(4));

for j=1:xSize(4)
    XiTrue = reshape(Xtrue(:,:,:,j),xSize(1),xSize(2),xSize(3));
    XiHat  = reshape(Xhat(:,:,:,j),xSize(1),xSize(2),xSize(3));
    SSIMArr(j) = ssim_index( rgb2gray(uint8(XiHat)), rgb2gray(uint8(XiTrue)));
end
SSIM = mean(SSIMArr);
end