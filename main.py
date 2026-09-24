import cv2 
import numpy as np 
import os 
import pickle 
from insightface.app import FaceAnalysis
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
def yolo_part(frame):
        
    results = model.track(frame, persist=True, classes=[0])[0]
    return results.boxes.id,results.boxes.xywh



app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320,320))

path = r"known faces" 
titles = os.listdir(path)

# Load existing file if it exists, otherwise start with empty lists 
if os.path.exists("trained_faces.pkl"): 
    with open("trained_faces.pkl","rb") as f: 
        list_encode,img_names = pickle.load(f) 
else: 
    list_encode = [] 
    img_names = []

# Look through folder but only process brand new images
updated = False 
for cls in titles: 
    name = os.path.splitext(cls)[0]
    if name not in img_names:
        curr_img = cv2.imread(f'{path}/{cls}')
        if curr_img is None:
            continue
        faces = app.get(curr_img)
        if len(faces) > 0:
            encode = faces[0].normed_embedding
            list_encode.append(encode)
            img_names.append(name)
            updated = True

# Save the list to your hard drive if a new image was added
if updated: 
    with open("trained_faces.pkl","wb") as f: 
        pickle.dump([list_encode,img_names],f)

cap = cv2.VideoCapture(0) 
process_this_frame = True 


THRESHOLD = 0.45
faces = []  # keep last detected faces available for skipped frames
kpx = 0.05
kpy = 0.05
confirm=0
while True: 
    success,frame = cap.read()  # always read, every loop, so the buffer never goes stale
    if not success:
        break

    if process_this_frame: 
        imgS = cv2.resize(frame,(0,0),None,fx=0.2,fy=0.2)
        faces = app.get(imgS)
    
    process_this_frame = not process_this_frame

    for face in faces:
        encoding = face.normed_embedding

        if len(list_encode) > 0:
            similarities = np.dot(list_encode,encoding)
            match_index = np.argmax(similarities)
            similarity = similarities[match_index]
        else:
            match_index = -1
            similarity = 0
            confirm=0

        x1,y1,x2,y2 = face.bbox.astype(int)
        x1,y1,x2,y2 = x1*5,y1*5,x2*5,y2*5

        if match_index >= 0 and similarity > THRESHOLD:
            confirm+=1
            if(confirm!=3):
                name = img_names[match_index].upper()

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame,(x1,y2+30),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(frame,name,(x1+6,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

                face_center_x = (x1+x2)//2
                face_center_y = (y1+y2)//2
                camera_center_x = frame.shape[1]//2
                camera_center_y = frame.shape[0]//2


                error_x = face_center_x-camera_center_x
                error_y = face_center_y-camera_center_y

                move_x = kpx * error_x
                move_y = kpy * error_y
                dead_zone = 50

                # Horizontal
                if error_x < -dead_zone:
                    horizontal = "LEFT"
                elif error_x > dead_zone:
                    horizontal = "RIGHT"
                else:
                    horizontal = "CENTER"

                # Vertical
                if error_y < -dead_zone:
                    vertical = "UP"
                elif error_y > dead_zone:
                    vertical = "DOWN"
                else:
                    
                    vertical = "CENTER"

                

                # Text
                cv2.putText(frame,horizontal,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                cv2.putText(frame,vertical,(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

                # Face Center
                cv2.circle(frame,(face_center_x,face_center_y),5,(0,0,255),-1)

                # Camera Center
                cv2.circle(frame,(camera_center_x,camera_center_y),5,(255,0,0),-1)

            else:
                name = "Unknown"
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.rectangle(frame,(x1,y2+30),(x2,y2),(0,0,255),cv2.FILLED)
                cv2.putText(frame,name,(x1+6,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else:
            yolo_part(frame)

    cv2.imshow("Webcam",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()