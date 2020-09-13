# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("./origin_img/1000209.png.png")


save = []
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        save.append((x, y))
        # print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

if __name__ == "__main__":
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey()

    print(save)
    # print('x:{}, y:{}'.format(save[-1]))
# while(True):
#     try:
#         cv2.waitKey()
#     except Exception:
#         cv2.destroyWindow("image")
#         break

