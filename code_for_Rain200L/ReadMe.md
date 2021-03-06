Requirements
========

Linux

pytorch

python3.6


Training
======

Download the training dataset from [Google Drive](https://drive.google.com/drive/folders/1u88Qz1duy5jMjUlpjWycXhSVqszo-0hF).

Unzip'rainy_image.zip', 'blurred_label.zip' and 'reference_clean_image.zip' in './datasets/train/'. 

Make sure the training images are in the'./datasets/train/rainy_image/' ,'./datasets/train/blurred_label/' and ./datasets/train/reference_clean_image/', respectively.

- Train the deraining model:

*python train.py --dataroot ./datasets/train/rainy_image/ --name new --model derain*


Testing
=======

Download the testing dataset from [Google Drive](https://drive.google.com/drive/folders/1UTCFU5wAxAJk2e3rtT6vmvOK41Jj_rEf).

Unzip'rainy_image.zip' and 'ground_truth.zip' in './datasets/test/'.

- Test:

*python test.py --dataroot ./datasets/test/rainy_image/ --name new --model derain*

- Test with our pretrained model:

*python test.py --dataroot ./datasets/test/rainy_image/ --name pretrained --model derain*

After the test, results are saved in './results/'.

Run "psnr_and_ssim.py" to caculate psnr and ssim.
