import cv2 
import numpy as np
from matplotlib import pyplot as plt
#main image
img_rgb = cv2.imread('C:/Users/halil/Desktop/luniz_project/squirt.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#shape templates, convert into grayscale
template_star = cv2.imread('C:/Users/halil/Desktop/luniz_project/star.jpg',0)
template_triangle = cv2.imread('C:/Users/halil/Desktop/luniz_project/triangle.jpg',0)
template_circle = cv2.imread('C:/Users/halil/Desktop/luniz_project/circle.jpg',0)
template_umbrella = cv2.imread('C:/Users/halil/Desktop/luniz_project/umbrella.jpg',0)
template_head = cv2.imread('C:/Users/halil/Desktop/luniz_project/head2.jpg',0)
template_head2 = cv2.imread('C:/Users/halil/Desktop/luniz_project/head.jpg',0)

w_star, h_star = template_triangle.shape[::-1]
w_triangle, h_triangle = template_triangle.shape[::-1]
w_circle, h_circle = template_triangle.shape[::-1]
w_umbrella, h_umbrella = template_triangle.shape[::-1]
w_head, h_head = template_head.shape[::-1]
w_head2, h_head2 = template_head2.shape[::-1]

threshold = 0.8 
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)


res_umbrella = cv2.matchTemplate(img_gray,template_umbrella,cv2.TM_CCOEFF_NORMED)
loc = np.where( res_umbrella >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_umbrella, pt[1] + h_umbrella), (0,255,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_umbrella, pt[1] + h_umbrella), (0,0,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)


res_circle = cv2.matchTemplate(img_gray,template_circle,cv2.TM_CCOEFF_NORMED)
loc = np.where( res_circle >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_circle, pt[1] + h_circle), (0,255,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_circle, pt[1] + h_circle), (0,0,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)



res_triangle = cv2.matchTemplate(img_gray,template_triangle,cv2.TM_CCOEFF_NORMED)
loc = np.where( res_triangle >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_triangle, pt[1] + h_triangle), (0,255,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_triangle, pt[1] + h_triangle), (0,0,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)




res_star = cv2.matchTemplate(img_gray, template_star,cv2.TM_CCOEFF_NORMED)
loc = np.where( res_star >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_star, pt[1] + h_star), (0,255,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_star, pt[1] + h_star), (0,0,0), 1)
cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(600)


res_head = cv2.matchTemplate(img_gray, template_head,cv2.TM_CCOEFF_NORMED)
loc = np.where( res_head >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_head, pt[1] + h_head), (0,255,0), 1)

res_head2 = cv2.matchTemplate(img_gray, template_head2,cv2.TM_CCOEFF_NORMED)
loc = np.where( res_head2 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w_head2, pt[1] + h_head2), (0,255,0), 1)

cv2.imshow("luniz_baba",img_rgb)
cv2.waitKey(4000)

