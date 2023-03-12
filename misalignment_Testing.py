#*******************************************************************************#
#       Copyright (c) 2022 TraceUp All Rights Reserved.			                #
# 	    Author: Kevin Foyet                                                     #
#*******************************************************************************#

# This Script will find the similarities between the right image and the left image provided by
# a camera and it will compare them to find the distance of the misalignment.
# For use on personal device make sure you install all libraries or the code won't work
from math import floor
from matplotlib import image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils

cameraID = input('Enter Camera ID: ') # Enter Camera ID for File detection, include the X before the id
SDA_NAME = "R"
SDB_NAME = "N"
LOCAL_PATH = Path(__file__).parent.joinpath("Imgs").absolute()

# Reads the 2 images at the designated file name locaction
img1 = cv.imread(str(Path(LOCAL_PATH).joinpath(cameraID, SDA_NAME + ".jpeg")))
img2 = cv.imread(str(Path(LOCAL_PATH).joinpath(cameraID, SDB_NAME + ".jpeg")))
print(str(Path(LOCAL_PATH).joinpath(cameraID, SDA_NAME + ".jpeg")))
print(str(Path(LOCAL_PATH).joinpath(cameraID, SDB_NAME + ".jpeg")))

# Cut the image in half
print(img1.shape)
print(img2.shape)
im = img1[ : , 2300: -1]
im2 = img2[: , :400]

# Saves the cropped image in the folder of the Id'ed camera
cv.imwrite(str(Path(LOCAL_PATH).joinpath(cameraID, "CroppedImg1" + ".jpeg")), im)
cv.imwrite(str(Path(LOCAL_PATH).joinpath(cameraID, "CroppedImg2" + ".jpeg")), im2)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(im,None)
kp2, des2 = sift.detectAndCompute(im2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=500)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
difference = 0
counter = 0
for i, (m1,m2) in enumerate(matches):
	if m1.distance < 0.7 * m2.distance:
		matchesMask[i] = [1,0]
		## Notice: How to get the index
		pt1 = kp1[m1.queryIdx].pt
		pt2 = kp2[m1.trainIdx].pt
		difference += abs((pt1[1] - pt2[1]))
		counter += 1
		print(i, pt1, pt2)
		print("average: ", abs(pt1[1] - pt2[1]))
		if i % 5 ==0:
			## Draw pairs in purple, to make sure the result is ok
			cv.circle(img1, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
			cv.circle(img2, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)
draw_params = dict(matchColor = (0,255,0),
				   singlePointColor = (255,0,0),
				   matchesMask = matchesMask,
				   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(im,kp1,im2,kp2,matches,None,**draw_params)
avg_misalignment_pixel = difference / counter
print("difference of y1 + y2: " , difference)
print("Number of iterations: ", counter)
print("Average misalignment pixel: ", avg_misalignment_pixel)


# Cropp Image number 3 to focus just on the middle of the image
cv.imwrite(str(Path(LOCAL_PATH).joinpath(cameraID, "Final Image" + ".jpeg")), img3)

#****************************************************************************************************#
#  Code Section for distance between object in the image to have a general idea of the misalignment  #
#****************************************************************************************************#
def midpoint(point_A, point_B):
	return ((point_A[0] + point_B[0])*0.5, (point_A[1] + point_B[1])*0.5)

# construct the argument such that one is the path of the image we want to measure and width of the image
final_image = str(cv.imread(str(Path(LOCAL_PATH).joinpath(cameraID, "Final Image" + ".jpeg"))))
argument_Parse = argparse.ArgumentParser()
argument_Parse.add_argument("-i", "--image", help = "path to the input image", default=r"C:\Users\Kevin\OneDrive - Cal Poly Pomona\Documents\GitHub\hardware_scripts\Camera\Imgs\X5123\Final Image.jpeg")
argument_Parse.add_argument("-w", "--width", type=float, help = "width of the left-most object in the image (in inches)", default=1)
args = vars(argument_Parse.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image_ = cv.imread(args["image"])
gray = cv.cvtColor(image_, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv.Canny(gray, 100, 200)
edged = cv.dilate(edged, None, iterations=1)
edged = cv.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and, then initialize the
# distance colors and reference object
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
	(255, 0, 255))
refObj = None

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	box = cv.minAreaRect(c)
	box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	# compute the center of the bounding box
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])

	# if this is the first contour we are examining (i.e.,
	# the left-most contour), we presume this is the
	# reference object
	if refObj is None:
		# unpack the ordered bounding box, then compute the
		# midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-right and
		# bottom-right
		(tl, tr, br, bl) = box
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		# compute the Euclidean distance between the midpoints,
		# then construct the reference object
		D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		refObj = (box, (cX, cY), D / args["width"])
		continue

	# draw the contours on the image
	orig = image_.copy()
	cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	# stack the reference coordinates and the object coordinates
	# to include the object center
	refCoords = np.vstack([refObj[0], refObj[1]])
	objCoords = np.vstack([box, (cX, cY)])

	# loop over the original points
	for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
		# draw circles corresponding to the current points and
		# connect them with a line
		cv.circle(orig, (int(xA), int(yA)), 5, color, -1)
		cv.circle(orig, (int(xB), int(yB)), 5, color, -1)
		cv.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
			color, 2)

		# compute the Euclidean distance between the coordinates,
		# and then convert the distance in pixels to distance in
		# units
		D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
		(mX, mY) = midpoint((xA, yA), (xB, yB))
		cv.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
			cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

		# show the output image
cv.imshow("Image", img3)
cv.waitKey(0)