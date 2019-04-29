#!/usr/bin/env python3
import cv2
import numpy as np
import random

def adjustGamma(imagnitudee, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255
        for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(imagnitudee.astype(np.uint8), table.astype(np.uint8))

def main():
    cap = cv2.VideoCapture('speed_data/train.mp4')

    ret, base_frame = cap.read()
    base_gray = cv2.cvtColor(adjustGamma(base_frame, random.uniform(1.0, 1.5)),cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(base_frame) 
    hsv[...,1] = 255
    #[hue, saturation, value]
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

        print(f'frame:{i}\nlabel:{tl.readline().strip()}\nGamma Val:{gammaVal}\n')

        flow = cv2.calcOpticalFlowFarneback(base_gray,
            next_gray, flow=None, pyr_scale=0.5,
            levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)

        magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

        print(magnitude[180])

        hsv[...,0] = angle*180/np.pi/2
        hsv[...,2] = cv2.normalize(src=magnitude, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        print(hsv[...,2][180])


        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow(f'previous frame', base_gray[50:360])
        cv2.imshow(f'current frame', next_gray[50:360])
        cv2.imshow('opticalflow',rgb[50:360])
        cv2.moveWindow('previous frame', 0, 0)
        cv2.moveWindow('current frame', 680, 0)
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
