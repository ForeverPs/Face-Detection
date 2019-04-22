import os
import cv2
import csv
import sys
import time
import numpy as np

class face(object):

	def __init__(self):
		self.path = []
		self.index = []
		self.number = []

	def generate(self):
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		camera = cv2.VideoCapture(0)
		total = input('How many people : ')
		for i in range(int(total)):
			print('\nPeople ' + str(i+1) + ': ')
			self.path.append(input('please input your name : '))
			self.index.append(int(input('please input your label : ')))
			if not os.path.lexists(self.path[i]):
				os.mkdir(self.path[i])
			if os.listdir(self.path[i]):
				print(42 * '-' + '\n WARNING:\n Please delete the previous folder first\n' + 42 * '-')
				return False
			flag = 0
			while flag != 'y':
				flag = input('begin? y/n : ')
				time.sleep(1)
			count = 0
			while flag == 'y':
				ret, frame = camera.read()
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = face_cascade.detectMultiScale(gray, 1.1, 3)
				for (x,y,w,h) in faces:
					img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
					f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
					store = self.path[i] + '/' + str(count) + '.pgm'
					cv2.imwrite(store, f)
					self.number.append(self.index[i])
					print('\rStore Image :', count, end = '')
					count += 1
				cv2.imshow('camera', frame)
				if cv2.waitKey(int(1000/12))&0xff == ord('q'):
					print('\n')
					flag = 'n'
					break
		camera.release()
		cv2.destroyAllWindows()
		return True

	def read_images(self,sz=None):
		c = 0
		X = []
		for path in self.path:
			names = os.listdir(path)
			names.sort(key=lambda x:int(x[:-4]))
			for name in names:
				X.append(cv2.imread(path + '/' + name, cv2.IMREAD_GRAYSCALE))
		return X, self.number

	def detect(self):
		X,Y = self.read_images()
		Y = np.asarray(Y, dtype=np.int32)
		model = cv2.face.EigenFaceRecognizer_create()
		model.train(np.asarray(X), np.asarray(Y))
		face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
		camera = cv2.VideoCapture(0)
		while True:
			read, img = camera.read()
			faces = face_cascade.detectMultiScale(img, 1.3, 5)
			for (x, y, w, h) in faces:
				img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0),2)
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				roi = gray[x:x+w,y:y+h]
				roi = cv2.resize(roi, (200,200), interpolation=cv2.INTER_LINEAR)
				params = model.predict(roi)
				name = self.path[self.index.index(params[0])]
				cv2.putText(img, name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
			cv2.imshow('camera', img)
			if cv2.waitKey(int(1000/12))&0xff == ord('q'):
				break
		cv2.destroyAllWindows()

if __name__ == '__main__':
	f = face()
	flag = f.generate()
	if flag:
		f.detect()
