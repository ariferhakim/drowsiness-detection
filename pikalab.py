# python pikalab.py

from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from functools import reduce
from gpiozero import Buzzer
from time import sleep
import numpy as np
import imutils
import time
import dlib
import cv2


def make_array_mata():
	A2 = np.zeros(shape=(1,2))
	B2 = np.zeros(shape=(1,2))
	C2 = np.zeros(shape=(1,2))
	A5 = [1,1,1,1,1]
	B5 = [1,1,1,1,1]
	C5 = [1,1,1,1,1]
	return A2,B2,C2,A5,B5,C5

def make_array_mulut():
	A5 = np.zeros(shape=(1,5))
	B5 = np.zeros(shape=(1,5))
	C5 = np.zeros(shape=(1,5))
	return A5,B5,C5

def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)

def input_array(A,B,C,AA,BB,CC):#untuk input ABC ke matriks 1x2 agar bisa dihitung max value
	X = np.append(A, AA)
	X = np.delete(X, 2, axis=0)
	Y = np.append(B, BB)
	Y = np.delete(Y, 2, axis=0)
	Z = np.append(C, CC)
	Z = np.delete(Z, 2, axis=0)
	return X,Y,Z

def normalisasi(AA,BB,CC,Ax,Bx,Cx):
	normA = AA[0]/Ax
	normB = BB[0]/Bx
	normC = CC[0]/Cx
	return normA,normB,normC

def mata(eye,AA,BB,CC):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	C = euclidean_dist(eye[0], eye[3])
	A,B,C = input_array(A,B,C,AA,BB,CC)
	return A,B,C

def max_value(A,B,C,Ax,Bx,Cx):
	Ax = Ax
	AAx = max(A)
	if AAx >= Ax:
		Ax = AAx
	Bx = Bx
	BBx = max(B)
	if BBx >= Bx:
		Bx = BBx
	Cx = Cx
	CCx = max(C)
	if CCx >= Cx:
		Cx = CCx
	return Ax,Bx,Cx

def tapis_mata(A,B,C,arrA,arrB,arrC,avA,avB,avC): #normalisasi, array, average
	aA = np.append(A,arrA)
	aA = np.delete(aA, 5, axis=0)
	if aA[4] != 1:
		avA = average(aA)
	aB = np.append(B,arrB)
	aB = np.delete(aB, 5, axis=0)
	if aB[4] != 1:
		avB = average(aB)
	aC = np.append(C,arrC)
	aC = np.delete(aC, 5, axis=0)
	if aC[4] != 1:
		avC = average(aC)
	return avA,avB,avC,aA,aB,aC

def eye_aspect_ratio(lavA,lavB,lavC,ravA,ravB,ravC):
	lear = (lavA+lavB)/(2.0*lavC)
	rear = (ravA+ravB)/(2.0*ravC)
	ear = (lear+rear)/2.0
	return ear

def average(values):
	return reduce(lambda a, b: a+b, values)/len(values)

def mulut(mouth):
	Am = euclidean_dist(mouth[2], mouth[10])
	Bm = euclidean_dist(mouth[4], mouth[8])
	Cm = euclidean_dist(mouth[0], mouth[6])

	return Am,Bm,Cm

def tapis_mulut(A,B,C,arrA,arrB,arrC,Amav,Bmav,Cmav):
	aA = np.append(A,arrA)
	aA = np.delete(aA, 5, axis=0)
	if aA[4] != 0:
		Amav = average(aA)
	aB = np.append(B,arrB)
	aB = np.delete(aB, 5, axis=0)
	if aB[4] != 0:
		Bmav = average(aB)
	aC = np.append(C,arrC)
	aC = np.delete(aC, 5, axis=0)
	if aC[4] != 0:
		Cmav = average(aC)
	return Amav,Bmav,Cmav,aA,aB,aC

def mouth_aspect_ratio(Am,Bm,Cm):
	if Am == 0 and Bm == 0 and Cm == 0:
		mar = 0
	else:
		mar = (Am + Bm)/(2.0 * Cm)
	return mar,Am,Bm,Cm

cascade = "haarcascade_frontalface_default.xml"
shape_predictor = "shape_predictor_68_face_landmarks.dat"
buzzer = Buzzer(17)
EYE_AR_THRESH = 0.7
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.76
MOUTH_AR_CONSEC_FRAMES = 5
nguap = 0
ngantuk = 0
Ax,Bx,Cx = 0,0,0
ravA,ravB,ravC,lavA,lavB,lavC = 1,1,1,1,1,1
lAx,lBx,lCx,rAx,rBx,rCx = 0,0,0,0,0,0
Amav,Bmav,Cmav = 0,0,0

COUNT = 0
COUNTER = 0
COUNT2 = 0 
ALARM_ON = False

print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier(cascade)
predictor = dlib.shape_predictor(shape_predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mulutStart, mulutEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# make array
la2,lb2,lc2,la5,lb5,lc5 = make_array_mata()
ra2,rb2,rc2,ra5,rb5,rc5 = make_array_mata()
ma5,mb5,mc5 = make_array_mulut()

# start the video stream thread
print("[INFO] starting video stream thread...")

vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=320)#ABAIKAN JIKA INGIN WRITE
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=25, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over face detections
	for (x, y, w, h) in rects:

		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #GAMBAR KOTAK DI FRAME
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# EKSTRAK KOORDINAT
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		lambe = shape[mulutStart:mulutEnd]
				
		#EAR
		leftA,leftB,leftC = mata(leftEye,la2,lb2,lc2)
		rightA,rightB,rightC = mata(rightEye,ra2,rb2,rc2)
		lAx,lBx,lCx = max_value(leftA,leftB,leftC,lAx,lBx,lCx)
		rAx,rBx,rCx = max_value(rightA,rightB,rightC,rAx,rBx,rCx)
		lnA,lnB,lnC = normalisasi(leftA,leftB,leftC,lAx,lBx,lCx)
		rnA,rnB,rnC = normalisasi(rightA,rightB,rightC,rAx,rBx,rCx)
		lavA,lavB,lavC,la5,lb5,lc5 = tapis_mata(lnA,lnB,lnC,la5,lb5,lc5,lavA,lavB,lavC) #normalisasi,array zeros 5, lav
		ravA,ravB,ravC,ra5,rb5,rc5 = tapis_mata(rnA,rnB,rnC,ra5,rb5,rc5,ravA,ravB,ravC) #mormalisasi, array zeros 5, rav
		ear = eye_aspect_ratio(lavA,lavB,lavC,ravA,ravB,ravC)
		
		#MAR
		Am,Bm,Cm = mulut(lambe)
		Amav,Bmav,Cmav,ma5,mb5,mc5 = tapis_mulut(Am,Bm,Cm,ma5,mb5,mc5,Amav,Bmav,Cmav)
		mar,Amulut,Bmulut,Cmulut = mouth_aspect_ratio(Amav,Bmav,Cmav)
		
		# DECISION MAKING
		if mar > MOUTH_AR_THRESH: #jika ear < threshold
			COUNT += 1 #untuk mengetahui durasi mata merem
			EYE_AR_THRESH = 0.01

			if COUNT >= MOUTH_AR_CONSEC_FRAMES: #jiika mata merem > THframe, alarm ON

				t = Thread(buzzer.on())
				t.deamon = True
				t.start()
									
				cv2.putText(frame, "BUKA MULUT", (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			buzzer.off()
			EYE_AR_THRESH = 0.7

		if ear < EYE_AR_THRESH: #jika ear < threshold
			COUNTER += 1 #untuk mengetahui durasi mata tutup

			if COUNTER >= EYE_AR_CONSEC_FRAMES: #jiika mata merem > THframe, alarm ON
				t = Thread(buzzer.on())
				t.deamon = True
				t.start()
					#buzzer.on()
				
				cv2.putText(frame, "MATA TUTUP", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			buzzer.off()

		cv2.putText(frame, "EAR: {:.3f}".format(ear), (100, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.3f}".format(mar), (100, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()

