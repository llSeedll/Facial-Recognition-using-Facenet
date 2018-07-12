import cv2
import numpy as np
import dlib
import glob
import os
import re
import pyautogui
from Keyboard import Keyboard
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *

import time

from Quartz import CGSessionCopyCurrentDictionary

password = '[PASSWORD]'
keyboard = Keyboard(password)

detector = dlib.get_frontal_face_detector()

FRmodel = load_model('face-rec.h5')
print("Total Params:", FRmodel.count_params())
predictor = dlib.shape_predictor("shape_predict_68_face_landmarks.dat")
thresh = 0.15


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def recognize_face(face_descriptor, database, names, dist_thresh, cond=False):
    encoding = img_to_encoding(face_descriptor, FRmodel)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
            break

    if cond and identity is not None and min_dist < dist_thresh:
        # If the user is identified
        name = names[identity]
        print('distance for %s is %s' % (name, dist))
        if name == "ADMIN":
            # If the identified user is "ADMIN"
            # Get the session's current state
            d = CGSessionCopyCurrentDictionary()
            is_locked = d and d.get("CGSSessionScreenIsLocked", 0) == 1
            if is_locked:
                # If the session is locked, try to unlock using the provided password
                print("locked: Trying to unlock---")
                time.sleep(1)
                pyautogui.click()
                keyboard.TypeUnicode(password)
                pyautogui.press('enter')
        return name, min_dist
    else:
        return None, None


def extract_face_info(img, img_rgb, database, names, ear, dist_thresh=.1):
    faces = detector(img_rgb)
    if len(faces) > 0:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            eyes_open = True if ear > thresh else False
            try:
                name, min_dist = recognize_face(image, database, names, dist_thresh, eyes_open)
            except Exception as e:
                print(e)
                continue

            if eyes_open:
                if min_dist < dist_thresh:
                    if name is None:
                        continue
                    cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            else:
                cv2.putText(img, 'Eyes Closed', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

def initialize():
    #load_weights_from_FaceNet(FRmodel)
    #we are loading model from keras hence we won't use the above method
    database = {}
    names = {}
    regex = re.compile(r'(.+)#(.+)')

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        if (file is not None):
            identity = os.path.splitext(os.path.basename(file))[0]
            for f in glob.glob(file + "/*"):
                if (f is not None):
                    database[identity] = fr_utils.img_path_to_encoding(f, FRmodel)
                    m = regex.match(os.path.splitext(os.path.basename(f))[0])
                    names[identity] = m.group(1)
                    #break

    return database, names


def recognize():
    database, names = initialize()
    cap = cv2.VideoCapture(0)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    while True:
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            #leftEyeHull = cv2.convexHull(leftEye)
            #rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
            extract_face_info(img, img_rgb, database, names, ear)
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


recognize()

