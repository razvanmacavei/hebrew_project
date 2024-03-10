import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

img = cv2.imread('D:\Downloaded Apps\Modern_Hebrew\Writer_1\Alphabet\\1_Alef\\11.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thresholding(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    plt.imshow(thresh, cmap='gray')
    return thresh

thresh_img = thresholding(img)
kernel = np.ones((3,1), np.uint8)
dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
plt.imshow(dilated, cmap='gray')

(contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1])
img2 = img.copy()


for ctr in sorted_contours_lines:
    
    x,y,w,h = cv2.boundingRect(ctr)
    cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 1)
    
    
plt.imshow(img2)

for _ in dilated:
    myArray = array(img)
    if myArray.any():
        cv2.imwrite("C:\\Users\\Razvan-PC\\OneDrive\\Desktop\\text1_{}.jpg".format((img)),thresh_img[y:y+h,x:x+w])
    

    



