import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1_path = '/media/iap205/Data4T/Export_temp/landmarks_view/202510/2c362cf42c944b61.jpg'
# img2_path = '/media/iap205/Data4T/Export_temp/landmarks_view/202510/46aa9ff68db81500.jpg'
# img2_path = '/media/iap205/Data4T/Export_temp/landmarks_view/202510/01d598aa2ee31917.jpg'
img2_path = '/media/iap205/Data4T/Export_temp/landmarks_view/202510/a9e3763587a3118a.jpg'

img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE) # trainImage
# img2 = cv.resize(img2, None,fx=0.7, fy=0.7)
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        matchesMask[i] = [1, 0]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
T, inliers = cv.estimateAffine2D(pts1,pts2)
inlier_num = inliers.sum()
print('inliers num: {}'.format(inlier_num))

inliers_matchesMask = matchesMask[:]
outliers_matchesMask = matchesMask[:]
j = 0
for i in range(len(matchesMask)):
    if 1 == matchesMask[i][0]:
        if 1 == inliers[j]:
            outliers_matchesMask[i] = [0, 0]
        else:
            inliers_matchesMask[i] = [0, 0]
        j += 1

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img5 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img5)
plt.axis('off')
plt.title('SIFT matches')
plt.title('All matches')
plt.show()
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = inliers_matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3= cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3)
plt.title('SIFT inliers')
plt.title('SIFT correspondences')
plt.title('Inliers')
plt.axis('off')
plt.show()
draw_params = dict(matchColor = (255,0,0),
                   singlePointColor = (255,0,0),
                   matchesMask = outliers_matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img4 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img4)
plt.axis('off')
plt.title('SIFT outliers')
plt.title('Outliers')
plt.show()


