import cv2
import imutils
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')


def distancehere(framepoint):
    varcount = 0
    p = []
    finalframe=[]
    for x,y,w,h in framepoint:
        p.append([x+w/2,y+h/2])
    #print(p)
    for i in p:
        dist = []
        for j in p:
            if j != i:
                dist.append(abs((i[0]-j[0])**2 + (i[1]-j[1])**2))
        print(dist)
        for j in dist:
            varcount = 0
            if j < 7000:
                finalframe.append(1)
                varcount = 1
                break
        if varcount == 0:    
           finalframe.append(0)
    #print(finalframe)


    return finalframe

def DifferenceFrame(frame1,frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(blur, None, iterations=1)
    #cv2.imshow('Motion Detected O/P',thresh)
    return dilated

def showHuman(bounding_box_cordinates,printableFrame):
    person = 1
    z = distancehere(bounding_box_cordinates)
    count = -1
    for x,y,w,h in bounding_box_cordinates: 
        count = count + 1
        if len(z) >=1:
            if z[count] == 0 :
                cv2.rectangle(printableFrame, (x,y), (x+w-10,y+h-10), (0,255,0), 2)
                cv2.putText(printableFrame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            elif z[count] == 1:
                cv2.rectangle(printableFrame, (x,y), (x+w-10,y+h-10), (0,0,255), 2)
                cv2.putText(printableFrame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            person += 1
    
    cv2.putText(printableFrame, 'Status : Detecting ', (550,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(printableFrame, f'Total Persons : {person-1}', (550,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    if person-1 >=7:
        cv2.putText(printableFrame, "LIMIT EXCEEDED",(550,100),cv2.FONT_ITALIC, 1,(0,0,255),4)
    cv2.imshow('output', printableFrame)
    #writer.write(frame)

    return printableFrame


def detect(frame):
    #cv2.imshow('Human Detected',frame)
    bounding_box_coordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    return (bounding_box_coordinates) #[[X1,Y1,W1,H1][X2,Y2,W2,H2]] 
    #print(bounding_box_cordinates,weights)

def humanDetector(args):
    
    video_path = args['video']
    if str(args["camera"]) == 'true' : camera = True 
    else : camera = False

    writer = None
    

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(ouput_path,writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, writer)

def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame1 = video.read()
        check, frame2 =video.read()
        frame = DifferenceFrame(frame1, frame2)
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByPathVideo(path, writer):
    writer = cv2.VideoWriter('output2.mp4',fourcc, 5, (1280,720))
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, frame1 =  video.read()
        check, frame2 = video.read()
        
        if check:
            frame1 = imutils.resize(frame1,width=min(800,frame.shape[1]))
            cv2.imshow('Input', frame1)
            frame2 = imutils.resize(frame2,width=min(800,frame.shape[1]))
            frame = DifferenceFrame(frame1, frame2)
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            BoxPoints = detect(frame)
            
            showHuman(BoxPoints, frame2)
            writer.write(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()




def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    humanDetector(args)
    