
## Train
python train.py --dataroot ./datasets/train/rainy_image/ --name new --model derain


# Test
python test.py --dataroot ./datasets/test/rainy_image/ --name new --model derain


# Test with our pretrained model
python test.py --dataroot ./datasets/test/rainy_image/ --name pretrained --model derain



######
After the test, results are saved in './results/'.

Run "psnr_and_ssim.py" to caculate psnr and ssim.
