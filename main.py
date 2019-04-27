import cv2 as cv
import numpy as np
import time

def main():
    #cv2.calcOpticalFlowPyrLK()
    cap = cv.VideoCapture('data/train.mp4')

    image_num = 1

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('end of video')
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow('frame',  frame[40:355])
        # print(image_num)
        # cv.imwrite(f'color_train.nosync/{image_num}.jpg', frame)
        # image_num += 1

        if cv.waitKey(1) == ord('q'):
            break

        time.sleep(0.05)
        

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()