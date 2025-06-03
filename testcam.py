from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if ret:
        print("frame read")
        results = model.predict(frame, show=True, conf=0.25, device="cpu")
        print("results", results)
    else:
        break

    # cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()