#!/usr/bin/env python3
import cv2
import numpy as np

def adjustGamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255
        for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def main():
    cap = cv2.VideoCapture('speed_data/train.mp4')

    ret, init = cap.read()
    init_gray = cv2.cvtColor(init,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(init)
    hsv[...,1] = 255

    tl = open('speed_data/train.txt', 'r')

    # while(True):
    #     cv2.imshow('HSV', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    #     if cv2.waitKey(1) == ord('q'):
    #         break
    i = 1
    while(1):
        ret, frame2 = cap.read()

        if not ret:
            print('end of video')
            break

        next = cv2.cvtColor(adjustGamma(frame2, 1.5),cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(init_gray,
            next, flow=None, pyr_scale=0.5,
            levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)

        print(tl.readline())
        # break

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', next[50:360])
        cv2.imshow('opticalflow',rgb[50:360])
        cv2.moveWindow('frame2', 0, 0)
        cv2.moveWindow('opticalflow', 0, 400)
        #cv2.imwrite(f'opFlow.nosync/opflow_{i}.png', rgb[50:360])

        

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        elif k == ord('q'):
            break
        init_gray = next
        i += 1
        
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
