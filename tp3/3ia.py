import cv2
import numpy as np

# Load the three input images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
img3 = cv2.imread('image3.jpg')

# Convert the input images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors for the first two images
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Find matches between the keypoints in the first two images
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des1, des2, k=2)

# Apply the ratio test to filter out bad matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Find the homography matrix for the stitched image and the second image
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Stitch the first two images together
stitched_img = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
stitched_img[0:img2.shape[0], 0:img2.shape[1]] = img2

# Find the keypoints and descriptors for the stitched image and the third image
gray_stitched = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
kp_stitched, des_stitched = sift.detectAndCompute(gray_stitched, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

# Find matches between the keypoints in the stitched image and the third image
matches = matcher.knnMatch(des_stitched, des3, k=2)

# Apply the ratio test to filter out bad matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Find the homography matrix for the stitched image and the third image
src_pts = np.float32([kp_stitched[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp3[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Stitch the third image to the stitched image of the first two images
result = cv2.warpPerspective(img3, H, (stitched_img.shape[1] + img3.shape[1], stitched_img.shape[0]))
result[0:stitched_img.shape[0], 0:stitched_img.shape[1]] = stitched_img

# Display the stitched image
cv2.imshow('Stitched Image', result)
