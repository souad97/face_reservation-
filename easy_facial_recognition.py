# Code Anis - Defend Intelligence
import cv2
import dlib
import PIL.Image
import numpy as np
import time
from imutils import face_utils
from text_to_speech import speak
import pyttsx3
#import RPi.GPIO as GPIO
engine = pyttsx3.init()

#RELAY = 17
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(RELAY, GPIO.OUT)
#GPIO.output(RELAY,GPIO.LOW)

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importing pretrained model..')


def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image,1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names):
    # Resize frame of video to 1/4 size for faster face recognition processing
    
    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
            engine.say("Welcome"+name)
            engine.runAndWait()
            print("door unlock")
            # to unlock the door
			#GPIO.output(RELAY,GPIO.HIGH)
			#doorUnlock = True
			
        else:
            name = "Unknown"
            engine.say("no place for you")
            engine.runAndWait()
            print("door locked")
            #doorUnlock = False
		   # GPIO.output(RELAY,GPIO.LOW)
            
            
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


if __name__ == '__main__':
    
    print('[INFO] Importing faces...')
    face_to_encode_path = ['known_faces/Zuckerberg.png','known_faces/souad.jpeg']
    known_face_encodings = []
    for face_to_encode_path in face_to_encode_path :
        image = PIL.Image.open(face_to_encode_path)
        image = np.array(image)
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)
        
    known_face_names=['Zuckerberg','souad']
    print('[INFO] Faces well imported')
    print('[INFO] Starting Webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam well started')
    print('[INFO] Detecting...')
    #doorUnlock = False
    while True:
        ret, frame = video_capture.read()
        easy_face_reco(frame, known_face_encodings, known_face_names)
        time.sleep(3)
        cv2.imshow('Easy Facial Recognition App', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Stopping System')
    video_capture.release()
    cv2.destroyAllWindows()
