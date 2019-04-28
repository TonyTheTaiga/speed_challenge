#!/usr/bin/env python3
import cv2
import numpy as np
import random

def adjustGamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255
        for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def main():
    cap = cv2.VideoCapture('speed_data/train.mp4')

    ret, base_frame = cap.read()
    base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(base_frame) 
    hsv[...,1] = 255

    tl = open('speed_data/train.txt', 'r')
    i = 1
    
    wk = 5

    while cap.isOpened():
        ret, next_frame = cap.read()

        if not ret:
            print('end of video')
            break

        gammaVal = random.uniform(1.0, 1.5)

        next_gray = cv2.cvtColor(adjustGamma(next_frame, gammaVal),cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(base_gray,
            next_gray, flow=None, pyr_scale=0.5,
            levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)

        print(f'frame:{i}\n{tl.readline()}\nGamma Val:{gammaVal}')

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow(f'current frame', next_gray[50:360])
        cv2.imshow('opticalflow',rgb[50:360])
        cv2.moveWindow('current frame', 0, 0)
        cv2.moveWindow('opticalflow', 0, 400)
        #cv2.imwrite(f'opFlow.nosync/opflow_{i}.png', rgb[50:360])

        base_gray = next_gray
        i += 1

        key = cv2.waitKey(wk) & 0xff
        if key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite('opticalfb.png',next_frame)
            cv2.imwrite('opticalhsv.png',rgb)
        elif key == ord('q'):
            break
        elif key == ord('p'):
            #pauses the video when p is pressed           
            if wk == 5:
                wk = 0
            else:
                wk = 5
            
        
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
