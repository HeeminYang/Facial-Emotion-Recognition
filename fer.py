import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Importing required packages
from keras.models import load_model
import keras
print('load model succeed')
#import tensorflow as tf
import numpy as np
import argparse
import dlib
import cv2
import json
import pandas as pd
import sys
from pathlib import Path


def NM(M, N):
    sorted_nums = sorted(M, reverse=True)
    k = sorted_nums[N]
    return k, list(M).index(k)

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

## 얼국 인식 영상 띄울때 쓰이는 부분
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

## 얼굴에 68개 랜드마크 찍는 모델 호출
faceLandmarks = "./faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

## 감정인식 모델 호출
emotionModelPath = './models/emotionModel.hdf5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
print('load model succeed')
emotionTargetSize = emotionClassifier.input_shape[1:3]

# Bring the '.mp4' file list in the folder and put it in videocapture
# file_path = os.listdir('/home/projects/Metaverse2021/Crop_Metaverse/higher/JWG/1session')

## 동영상 위치, 동일한 확장자 비디오 호출해서 감정인식
root_path = Path('/home/projects/Metaverse2021/DuckEEsStim/')
# root_path = Path('/home/projects/Metaverse2021/LIRISCSE/videos_208/')
file_paths = np.array(list(root_path.rglob('*.m4v')))
# file = 'LYG_2_30'
# file_paths = ['/home/projects/Metaverse2021/Crop_Emotion/'+file]
print(root_path)
print(file_paths)

for file_path in file_paths:
    print(file_path)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    cap = cv2.VideoCapture(file_path.as_posix())
    # cap = cv2.VideoCapture(file_path+'.mp4')
    # cap = cv2.VideoCapture("/home/projects/Metaverse2021/Crop_Metaverse/higher/JWG/1session/session1_3분전_JWG.mp4")
    # print(cv2.VideoCapture.read(cap))
    

    ## 동영상의 총 frame 수와 얼굴인식이 되는 frame수를 세야함
    totalcount = 0
    landcount = 0
    sll = []
    
    c = 0
    while True:
        ret, frame = cap.read()
    
        if frame is None:
            break
        try: 
            frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)
        #print(f"Count {c} | Reading frame succeed : {frame.shape}", end = ' | ')

        totalcount += 1
        cv2.putText(frame, 'Picture: '+str(totalcount), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #print('Putting text succeed', end = ' | ')
        # if not ret:
        #     break

        # grayFrame = frame
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print('Converting color succeed', end = ' | ')
        rects = detector(grayFrame, 0)
        #print('Detecting face succeed')
        c += 1
        for rect in rects:
            ## rects 얼굴 잡은 프레임들 (사각형으로 잡아서 rects)

            shape = predictor(grayFrame, rect)
            points = shapePoints(shape)
            
            landcount += 1
            cv2.putText(frame, 'Landmark: ' + str(landcount), (0,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            shape_list = []

            shape_list.append(totalcount)
            shape_list.append(landcount)
                ## append (x, y) in shape_list

            ## 68개 랜드마크도 x y 값을 따로 저장 (추후 df에)
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

            ## 감정 인식
            emotion_prediction = emotionClassifier.predict(grayFace)
            emotion_probability = np.max(emotion_prediction)

            ## 예측한 감정이 softmax 0.36을 넘으면 해당 감정으로 예측 확정
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
                softmax_emotion = emotion_prediction[0]

            ## 넘지 못하면 none = 인식 불가
            else:
                color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                emo = 'none'
                none = ['none','none','none','none','none','none','none']

            if 'softmax_emotion' in locals():
                shape_list.append(emo)
                shape_list.extend(softmax_emotion)
                sll.append(shape_list)
            else:
                shape_list.append(emo)
                shape_list.extend(none)
                sll.append(shape_list)
                
        cv2.putText(frame, 'Landmark: '+str(landcount), (0,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # if args["isVideoWriter"] == True:
        #     videoWrite.write(frame)

        # cv2.imshow("Emotion Recognition", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


    ## df 구성
    # 열이름 : total_count,landmark_count,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,emotion,Angry,Disgust,Fear,Happy,Sad,Suprise,Neutral
    coln = ['total_count', 'landmark_count']
    coln += list(range(136))
    coln += ['emotion', emotions[0]['emotion'], emotions[1]['emotion'], emotions[2]['emotion'], 
        emotions[3]['emotion'], emotions[4]['emotion'], emotions[5]['emotion'], emotions[6]['emotion']]
    df = pd.DataFrame(sll, columns=coln)

    df.to_csv('/home/heemin/mv/dir/DuckEEs/{}.csv'.format(file_path.stem), index=False)
    # df.to_csv('/home/heemin/mv/dir/liriscse/{}.csv'.format(file_path.stem), index=False)

    cap.release()
    # if args["isVideoWriter"] == True:
    #     videoWrite.release()
    cv2.destroyAllWindows()