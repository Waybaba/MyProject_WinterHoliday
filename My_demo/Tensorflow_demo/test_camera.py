import o
import numpy

cap = cv2.VideoCapture(1) # 调整参数实现读取视频或调用摄像头
while 1:
    ret, frame = cap.read()
    cv2.imshow("cap", frame)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()