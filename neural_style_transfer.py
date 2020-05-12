# USAGE
# python neural_style_transfer.py --image images/baden_baden.jpg --model models/instance_norm/starry_night.t7

# import the necessary packages
import argparse
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="models/instance_norm/the_scream.t7", required=False,
	help="neural style transfer model")
ap.add_argument("-i", "--image", default="images/face1.jpeg", required=False,
	help="input image to apply neural style transfer to")
args = vars(ap.parse_args())

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])

# load the input image, resize it to have a width of 600 pixels, and
# then grab the image dimensions
image = cv2.imread(args["image"])
# image = cv2.imread("images/face1.jpeg")
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image, set the input, and then perform a
# forward pass of the network
blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
	(103.939, 116.779, 123.680), swapRB=False, crop=False)
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()

# reshape the output tensor, add back in the mean subtraction, and
# then swap the channel ordering
output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680
output /= 255.0
output = output.transpose(1, 2, 0)

# show information on how long inference took
print("[INFO] neural style transfer took {:.4f} seconds".format(
	end - start))

# show the images

# output/=output.max()
# output*=255
# img = output.astype(np.uint8)

img = cv2.normalize(src=output, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("Input", image)
cv2.imshow("Output", img)
cv2.waitKey(0)

cv2.imwrite("images/output/output.jpg", img)