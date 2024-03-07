# Importing required packages
from keras.models import load_model 
import numpy as np
import argparse
import dlib
import cv2
import json
import pandas as pd
import os

direct_list =['/home/heemin/mv/video/1.MP4']

# controler
video_save = False

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=video_save)
args = vars(ap.parse_args())

# # create list for landmarks
# ALL = list(range(0, 68))
# RIGHT_EYEBROW = list(range(17, 22))
# LEFT_EYEBROW = list(range(22, 27))
# RIGHT_EYE = list(range(36, 42))
# LEFT_EYE = list(range(42, 48))
# NOSE = list(range(27, 36))
# MOUTH_OUTLINE = list(range(48, 61))
# MOUTH_INNER = list(range(61, 68))
# JAWLINE = list(range(0, 17))

emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}


def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = 'models/emotionModel.hdf5'  # anlj
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

for dir in direct_list:
    cap = cv2.VideoCapture(dir)
    # cap = cv2.VideoCapture(0)

    if args["isVideoWriter"] == True:
        fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        # capWidth = int(cap.get(3))
        # capHeight = int(cap.get(4))
        videoWrite = cv2.VideoWriter(dir + "_output.avi", fourrcc, 30,
                                    (720, 480))

    totalcount = 0
    landcount = 0
    sll=[]

    while True:
        ret, frame = cap.read()
        try: 
            frame = cv2.resize(frame.astype(np.uint8), (720, 480))
        except Exception as e:
            print(e)
            
        totalcount += 1
        
        cv2.putText(frame, 'Picture: '+str(totalcount), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not ret:
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(grayFrame, 0)
        for rect in rects:
            shape = predictor(grayFrame, rect)
            points = shapePoints(shape)
            
            landcount += 1
            cv2.putText(frame, 'Landmark: '+str(landcount), (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # face landmark code
            ## create list to contain shape
            shape_list = []

            shape_list.append(totalcount)
            shape_list.append(landcount)
            ## append (x, y) in shape_list
            for p in shape.parts():
                shape_list.append(p.x)
                shape_list.append(p.y)
                cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)
            
            (x, y, w, h) = rectPoints(rect)
            grayFace = grayFrame[y:y + h, x:x + w]
            try:
                grayFace = cv2.resize(grayFace, (emotionTargetSize))
            except:
                continue

            grayFace = grayFace.astype('float32')
            grayFace = grayFace / 255.0
            grayFace = (grayFace - 0.5) * 2.0
            grayFace = np.expand_dims(grayFace, 0)
            grayFace = np.expand_dims(grayFace, -1)
            emotion_prediction = emotionClassifier.predict(grayFace)
            emotion_probability = np.max(emotion_prediction)
            if (emotion_probability > 0.36):
                emotion_label_arg = np.argmax(emotion_prediction)
                color = emotions[emotion_label_arg]['color']
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                        color,
                        thickness=2)
                cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40),
                            color, -1)
                cv2.putText(frame, emotions[emotion_label_arg]['emotion'],
                            (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                emo = emotions[emotion_label_arg]['emotion']
            else:
                color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                emo = 'none'
            shape_list.append(emo)
            sll.append(shape_list)
            
            # # landmark value save as json
            # with open("test.json", "w") as json_file:
            #     key_val = [ALL, sll]
            #     landmark_dict = dict(zip(*key_val))
            #     json_file.write(json.dumps(landmark_dict))
            #     json_file.write('\n')
        cv2.putText(frame, 'Landmark: '+str(landcount), (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if args["isVideoWriter"] == True:
            videoWrite.write(frame)

        # cv2.imshow("Emotion Recognition", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # landmark value save as csv(DataFrame)
    coln = ['total_count', 'landmark_count']
    coln += list(range(136))
    coln += ['emotion']
    df = pd.DataFrame(sll, columns=coln)
    df.to_csv(dir+'.csv', index=False)

    cap.release()
    if args["isVideoWriter"] == True:
        videoWrite.release()
    cv2.destroyAllWindows()