#reference: https://pylessons.com/OpenCV-image-stiching-continue

import cv2
import numpy as np
#original_image_right
img_r = cv2.imread('image1.jpg')
imgr = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

#original_image_left
img_l = cv2.imread('image0.jpg')
imgl = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# find key points
kpr, desr = sift.detectAndCompute(imgr,None)
kpl, desl = sift.detectAndCompute(imgl,None)

cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_l,kpl,None))
cv2.waitKey(0)

cv2.imshow('original_image_right_keypoints',cv2.drawKeypoints(img_r,kpr,None))
cv2.waitKey(0)

match = cv2.BFMatcher()
matches = match.knnMatch(desr,desl,k=2)

good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)

draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

img3 = cv2.drawMatches(img_r,kpr,img_l,kpl,good,None,**draw_params)
cv2.imshow("Draw Matches Left Right.jpg", img3)
cv2.waitKey(0)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kpr[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpl[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h,w = imgr.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    imgr = cv2.polylines(imgr,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    cv2.imshow("right_image_overlapping.jpg", imgr)
    cv2.waitKey(0)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
print(img_l.shape[1] + img_r.shape[1], img_l.shape[0])
dst = cv2.warpPerspective(img_r,M,(img_l.shape[1] + img_r.shape[1], img_l.shape[0]))
cv2.imshow("dst warpPerspective", dst)
cv2.waitKey(0)

dst[0:img_l.shape[0],0:img_l.shape[1]] = img_l
cv2.imshow("add left image to the warped right.jpg", dst)
cv2.waitKey(0)

def trim(frame):
    #crop 
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop 
    if not np.sum(frame[-1]):
        return trim(frame[0:-1])
    #crop 
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop 
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
#https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[:, -1])
print(arr[:, :-2])
print(arr[:,0])
print(arr[:,1:])
print(arr[-1])
print(arr[0:-1])
print(arr[0])
print(arr[1:])

cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.imwrite("original_image_stitched_crop.jpg", trim(dst))
