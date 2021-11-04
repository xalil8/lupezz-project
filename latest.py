import cv2 
import numpy as np

#main image
img_rgb = cv2.imread('luniz_project/templates/squirt1.jpg')
template_star = cv2.imread('luniz_project/templates/star.jpg',0)
template_triangle = cv2.imread('luniz_project/templates/triangle.jpg',0)
template_circle = cv2.imread('luniz_project/templates/circle.jpg',0)
template_umbrella = cv2.imread('luniz_project/templates/umbrella.jpg',0)
template_head = cv2.imread('luniz_project/templates/head.jpg',0)
template_head2 = cv2.imread('luniz_project/templates/head2.jpg',0)

templates = [template_umbrella, template_circle, template_triangle, template_star, template_head2, template_head]

def template_match(image, template):
    w, h = template.shape[::-1]
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)

    return image
     

cv2.imshow("Screen", img_rgb)
cv2.waitKey(600)

for template in templates:
    raw_image = template_match(img_rgb, template)
    cv2.imshow("Screen", raw_image)
    cv2.waitKey(600)

cv2.imwrite()