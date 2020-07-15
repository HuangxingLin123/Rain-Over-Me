Download the training dataset from [Google Drive](https://drive.google.com/drive/folders/1Zhi3nYUdhfBFRpcJzvlBS3-XEgDsCfLc).

Unzip'rainy_image.zip', 'blurred_label.zip' and'reference_clean_image.zip' in './datasets/train/'. 

Make sure the training images are in the'./datasets/train/rainy_image/' ,'./datasets/train/blurred_label/' and ./datasets/train/reference_clean_image/', respectively.

- Train the deraining model:

python train.py --dataroot ./datasets/train/rainy_image/ --name new --model derain
