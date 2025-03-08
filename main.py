
import cv2 
import torch 
import numpy as np 

#do git clone https://github.com/WongKinYiu/yolov7.git 
#install dependencies pip install -r requirements.txt

model = torch.hub.load('ultralytics/yolov5', 'yolov5n') 

def detect_objects(): 
        cap = cv2.VideoCapture(0)
        objects = [] 

        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            results = model(frame)

            detected_objects = results.names  # Extract object labels
            predictions = results.pred[0]  # Predicted bounding boxes

            for pred in predictions:
                x1, y1, x2, y2, conf, class_id = pred  # Extract coordinates, confidence, and class
                if conf > 0.3:
                    object_label = detected_objects[int(class_id)]
                    objects.append(object_label)
            
            renderedframes = results.render()[0]

            # Display the resulting frame
            cv2.imshow('YOLOv5 Assistive Object Detection', renderedframes)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

detect_objects()





