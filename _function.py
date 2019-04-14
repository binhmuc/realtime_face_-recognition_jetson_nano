import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop_face(imgarray, section, margin=50, size=112):
	"""
	:param imgarray: full image
	:param section: face detected area (x, y, w, h)
	:param margin: add some margin to the face detected area to include a full head
	:param size: the result image resolution with be (size x size)
	:return: resized image in numpy array with shape (size x size x 3)
	"""
	img_h, img_w, _ = imgarray.shape
	if section is None:
		section = [0, 0, img_w, img_h]
	(x, y, w, h) = section
	margin = int(min(w,h) * margin / 100)
	x_a = x - margin
	y_a = y - margin
	x_b = x + w + margin
	y_b = y + h + margin
	if x_a < 0:
		x_b = min(x_b - x_a, img_w-1)
		x_a = 0
	if y_a < 0:
		y_b = min(y_b - y_a, img_h-1)
		y_a = 0
	if x_b > img_w:
		x_a = max(x_a - (x_b - img_w), 0)
		x_b = img_w
	if y_b > img_h:
		y_a = max(y_a - (y_b - img_h), 0)
		y_b = img_h
	cropped = imgarray[y_a: y_b, x_a: x_b]
	resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
	resized_img = np.array(resized_img)
	plt.imshow(resized_img)
	return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
		
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1, thickness=2):
	size = cv2.getTextSize(label, font, font_scale, thickness)[0]
	x, y = point
	cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
	cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
		