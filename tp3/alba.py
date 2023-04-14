import cv2
import numpy as np

# Load input images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
img3 = cv2.imread('image3.jpg')

# Convert input images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for all three images
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

# Match keypoints between image 1 and image 2
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Calculate homography between image 1 and image 2
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H1to2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Match keypoints between image 2 and image 3
matches = matcher.knnMatch(des2, des3, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Calculate homography between image 2 and image 3
if len(good_matches) > 10:
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp3[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H2to3, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Combine homographies to create transformation matrix for stitching all three images
H1to3 = np.dot(H2to3, H1to2)

# Warp image 1 and image 2 onto a common plane using homography matrix
img2_warped = cv2.warpPerspective(img2, H1to2, (img1.shape[1] + img2.shape[1], img1.shape[0]))
img3_warped = cv2.warpPerspective(img3, H1to3, (img1.shape[1] + img2.shape[1], img1.shape[0]))

# Combine warped images into a single panorama
panorama = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1] + img3.shape[1], 3), dtype=np.uint8)
panorama[0:img1.shape[0], 0:img1.shape[1], :] = img1
panorama[0:img2.shape[0], img1.shape[1]:img1.shape[1]]
