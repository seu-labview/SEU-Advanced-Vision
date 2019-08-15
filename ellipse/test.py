import numpy as np
import cv2
if __name__ == "__main__":
    # img = cv2.imread('1_depth_Depth.png')
    img = cv2.imread('depth_limited_color_image.jpg')
    # cv2.threshold(img, 30, 255, cv2.THRESH_BINARY, img)
    # print(img.size)
    cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('depth', img)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()