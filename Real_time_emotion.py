import sys
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
import cv2
from kivy.uix.screenmanager import ScreenManager, Screen 
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


Builder.load_string('''
<design>
    rows:3
    size: root.width, root.height
    Label:
        text: ('[color=00FF00][b]Real Time Emotion Recognition[/b][/color]')
        font_size: 35
        markup: True
        size_hint: [0.3,0.3]
        pos_hint: {"x":0.34, "top":1.05}
    Image:
        source: 'capture.png'
        size_hint: [0.5,0.5]
        pos_hint: {"x":0.25, "top":0.8}
    Button:
        text: "[color=a][b]Start[/b][/color]"
        markup: True
        width: 40
        font_size: 25
        size_hint: 0.3, 0.1
        pos_hint: {"x":0.35, "top":0.2} 
        background_color: (0.0, 2.0, 0.0, 1.0) 
        on_release: root.capture()    
''')

class design(Screen):
    def capture(self):

        model = model_from_json(open("new.json", "r").read())
        model.load_weights('new.h5')
        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap=cv2.VideoCapture(0)

        while True:
            ret,test_img=cap.read()# captures frame and returns boolean value and captured image
            if not ret:
                continue
            gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
            for (x,y,w,h) in faces_detected:
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
                roi_gray=cv2.resize(roi_gray,(48,48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                #find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('Facial emotion analysis ',resized_img)

            if cv2.waitKey(10) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows 
    


    

screen_manager = ScreenManager() 

screen_manager.add_widget(design(name ="screen_one")) 


class TestCamera(App):

    def build(self):
        return screen_manager


TestCamera().run()
