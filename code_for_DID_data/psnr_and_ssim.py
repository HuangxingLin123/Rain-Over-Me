import skimage
import cv2
from skimage.measure import compare_psnr, compare_ssim



def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    return compare_ssim(im1_y, im2_y)


## ground truth
dir='./datasets/test/ground_truth/'


##  derained results
dir4="./results/pretrained/test_latest/images/"


ssim=0
psnr=0

total_num=1200
for i in range(1,1201):

    img1= cv2.imread(dir+str(i)+'.jpg')
    (h, w, n) = img1.shape
    #
    # if h%8!=0 or w%8!=0:
    #     img1=cv2.resize(img1,(int(h/8)*8+8,int(w/8)*8+8))

    img2= cv2.imread(dir4+str(i)+'_Y_hat.png')

    a = calc_ssim(img1, img2)
    b = calc_psnr(img1, img2)
    print(i, ':', a, b)
    ssim += a
    psnr += b

ssim=ssim/total_num
psnr=psnr/total_num
print("ssim=",ssim)
print("psnr=",psnr)




