import cv2 
import numpy as np 
import os 
import pickle 
from insightface.app import FaceAnalysis
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def yolo_part(frame):
        
    results = model.track(frame, persist=True, classes=[0])[0]
    return results.boxes.id,results.boxes.xyxy


def load_known_faces(app):
    path = r"C:\Users\au773\Documents\VS Code\FaceRecognition\knownfaces" 
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

    return list_encode, img_names


def get_face_box(face):
    x1,y1,x2,y2 = face.bbox.astype(int)
    x1,y1,x2,y2 = x1*5,y1*5,x2*5,y2*5
    return x1,y1,x2,y2


def match_face(list_encode, encoding):
    if len(list_encode) > 0:
        similarities = np.dot(list_encode,encoding)
        match_index = np.argmax(similarities)
        similarity = similarities[match_index]
    else:
        match_index = -1
        similarity = 0
        confirm=0
    return match_index, similarity


def draw_known_face(frame,x1,y1,x2,y2,name):
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(frame,(x1,y2+30),(x2,y2),(0,255,0),cv2.FILLED)
    cv2.putText(frame,name,(x1+6,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


def draw_unknown_face(frame,x1,y1,x2,y2,name):
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.rectangle(frame,(x1,y2+30),(x2,y2),(0,0,255),cv2.FILLED)
    cv2.putText(frame,name,(x1+6,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


def track_face_position(frame,x1,y1,x2,y2,kpx,kpy):
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


def find_track_id_at_point(ids, boxes, px, py):
    best_id, best_dist = None, float("inf")
    for track_id, box in zip(ids, boxes):
        x1b,y1b,x2b,y2b = box
        if x1b <= px <= x2b and y1b <= py <= y2b:
            return int(track_id)
        cx = (x1b+x2b)/2
        cy = (y1b+y2b)/2
        dist = np.hypot(cx-px, cy-py)
        
        if dist < best_dist:
            best_dist, best_id = dist, int(track_id)
    return best_id


def draw_locked_body(frame, ids, boxes, locked_id, kpx, kpy):
    if ids is None:
        return False
    for track_id, box in zip(ids, boxes):
        if int(track_id) == locked_id:
            x1b,y1b,x2b,y2b = box
            x1b = int(x1b)
            y1b = int(y1b)
            x2b = int(x2b)
            y2b = int(y2b)
            cv2.rectangle(frame,(x1b,y1b),(x2b,y2b),(0,255,255),2)
            cv2.putText(frame,"LOCKED",(x1b,y1b-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            track_face_position(frame,x1b,y1b,x2b,y2b,kpx,kpy)
            return True
    return False


def run_pipeline(app, list_encode, img_names):
    cap = cv2.VideoCapture(0) 
    process_this_frame = True 


    THRESHOLD = 0.45
    faces = []  # keep last detected faces available for skipped frames
    kpx = 0.05
    kpy = 0.05
    confirm=0
    prev_name=None
    locked=False
    locked_id=None
    while True: 
        success,frame = cap.read()  # always read, every loop, so the buffer never goes stale
        if not success:
            break

        if not locked:
            if process_this_frame: 
                imgS = cv2.resize(frame,(0,0),None,fx=0.2,fy=0.2)
                faces = app.get(imgS)
            
            process_this_frame = not process_this_frame

            for face in faces:
                encoding = face.normed_embedding

                match_index, similarity = match_face(list_encode, encoding)

                x1,y1,x2,y2 = get_face_box(face)

                if match_index >= 0 and similarity > THRESHOLD:
                    name = img_names[match_index].upper()

                    if name == prev_name:
                        confirm+=1
                    else:
                        confirm=1
                        prev_name=name

                    if confirm!=3:
                        draw_known_face(frame,x1,y1,x2,y2,name)
                        track_face_position(frame,x1,y1,x2,y2,kpx,kpy)
                    else:
                        face_center_x = (x1+x2)//2
                        face_center_y = (y1+y2)//2
                        ids, boxes = yolo_part(frame)
                        if ids is not None:
                            locked_id = find_track_id_at_point(ids, boxes, face_center_x, face_center_y)
                            if locked_id is not None:
                                locked=True
                else:
                    name = "Unknown"
                    confirm=0
                    prev_name=None
                    draw_unknown_face(frame,x1,y1,x2,y2,name)

        else:
            ids, boxes = yolo_part(frame)
            found = draw_locked_body(frame, ids, boxes, locked_id, kpx, kpy)
            if not found:
                locked=False
                locked_id=None
                confirm=0
                prev_name=None

        cv2.imshow("Webcam",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(320,320))

    list_encode, img_names = load_known_faces(app)
    run_pipeline(app, list_encode, img_names)


if __name__ == "__main__":
    main()