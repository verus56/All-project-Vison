import numpy as np
import cv2
import os

# Directory containing the query and train images
query_dir = "C:/Users/Thameur/Desktop/miv/vision/CV_TP2/queries"
train_dir = "C:/Users/Thameur/Desktop/miv/vision/CV_TP2/License Plates"
train_dir

# Get all the query image files in the directory
query_files = [os.path.join(query_dir, f) for f in os.listdir(query_dir) if f.endswith(".jpg") or f.endswith(".png")]

# Get all the train image files in the directory
train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".jpg") or f.endswith(".png")]

# Read the query images
query_images = []
for query_file in query_files:
    query_img = cv2.imread(query_file, 0)
    query_images.append(query_img)

# Read the train images
train_images = []
for train_file in train_files:
    train_img = cv2.imread(train_file, 0)
    train_images.append(train_img)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors for each train image
train_keypoints = []
train_descriptors = []
for train_img in train_images:
    kp, des = sift.detectAndCompute(train_img, None)
    train_keypoints.append(kp)
    train_descriptors.append(des)

# Match descriptors between query and train images
bf = cv2.BFMatcher()
for i, query_img in enumerate(query_images):
    query_kp, query_des = sift.detectAndCompute(query_img, None)
    best_match = None
    best_match_count = 0
    for j, train_img in enumerate(train_images):
        matches = bf.knnMatch(query_des, train_descriptors[j], k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        if len(good) > best_match_count:
            best_match = train_img
            best_match_count = len(good)
            best_matches = good

    # Draw matching keypoints between query and train images
    img3 = cv2.drawMatchesKnn(query_img, query_kp, best_match, train_keypoints[j], best_matches, None, flags=2)
    cv2.imshow("Best match for query image {}".format(i), img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
