import cv2 as cv
img = cv.imread('im1.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg',img)
cv.imshow('sift_keypoints',img)
cv.waitKey(0)

detection les pts point sifte dans 2 image
calclude les donne 3d de petite movment