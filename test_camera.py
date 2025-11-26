import sys
sys.path.append('/home/erlin/work/labgrasp')
from create_camera import CreateRealsense
import cv2

# camera = CreateRealsense("231522072272") #
camera = CreateRealsense("250122078799") #


while True:
    color = camera.get_frames()['color']
    cv2.imshow('color image', color)
    
    key = cv2.waitKey(1)  # 等待1ms，刷新窗口
    if key == ord('q'):   # 按 q 退出
        break

cv2.destroyAllWindows()