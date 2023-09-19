import cv2 as cv

def capture():
	capture = cv.VideoCapture(0)
	ret, image = capture.read()
	if ret is True:
		image = cv.flip(image, 1)
		#cv.imshow( "frame", image)
		#保存图片到桌面
		cv.imwrite("/home/zyp/Desktop/image/1.jpg", image)

