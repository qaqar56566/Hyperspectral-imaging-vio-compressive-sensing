close all

figure;
imshow(compressed_img);
 title("compressed image");

for bShow = 1:31
    figure;
    subplot(1,3,1); imshow(HS(:,:,bShow), []); 
    title(sprintf("GT HS band %d", bShow));

    subplot(1,3,2); imshow(recon_cube(:,:,bShow), []); 
    title(sprintf("TwIST+TV recon band %d", bShow));


    subplot(1,3,3); imshow(recon_cube(:,:,bShow)-HS(:,:,bShow), []); 
    title(sprintf("TwIST+TV recon band %d", bShow));

end


% recons = load('tau_sweep_results\best_recon_so_far.mat');
