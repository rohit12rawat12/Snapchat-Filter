import cv2
import numpy
import dlib
import math

cap = cv2.VideoCapture(0)
nose_image = cv2.imread("image/pig img.png")
pi = 22/7;

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = detector(gray_frame)
	for face in faces:

		landmarks = predictor(gray_frame, face)
		#print(landmarks)
		top_nose = (landmarks.part(29).x, landmarks.part(29).y)
		center_nose = (landmarks.part(30).x, landmarks.part(30).y)
		left_nose = (landmarks.part(31).x, landmarks.part(31).y)
		right_nose = (landmarks.part(35).x, landmarks.part(35).y)
		
		# calculating nose_width using distance formula 
		nose_width = int((((left_nose[0] - right_nose[0])**2
		 		+ (left_nose[1] - right_nose[1])**2)**0.5)*1.7) #1.7 to increase nose size
		
		nose_height = int(nose_width * 0.819) #0.819 is the ratio of image height to image width

		# new nose position
		top_left = (int(center_nose[0] - nose_width/2), int(center_nose[1] - nose_height/2))
		bottom_right =(int(center_nose[0] + nose_width/2), int(center_nose[1] + nose_height/2)) 
		
		# Adding new nose
		nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
		nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
		_, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
		nose_area = frame[top_left[1] : top_left[1] + nose_height,
					top_left[0] : top_left[0] + nose_width]

		nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
		final_nose = cv2.add(nose_area_no_nose, nose_pig)
		
		frame[top_left[1] : top_left[1] + nose_height, 
			  top_left[0] : top_left[0] + nose_width] = final_nose

		#cv2.imshow("nose area", nose_area)
		#cv2.imshow("pig nose", nose_pig)
		#cv2.imshow("gray pig nose", nose_pig_gray)
		#cv2.imshow("mask", nose_mask)
		#cv2.imshow("final nose", final_nose)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
