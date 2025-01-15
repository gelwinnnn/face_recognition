import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from types import MethodType

### Helper Function
def encode(img):
    """
    Generate embeddings using the ResNet model.
    """
    with torch.no_grad():
        img_embedding = resnet(img)
    return img_embedding

def detect_box(self, img, save_path=None):
    """
    Detect faces and extract face crops using MTCNN.
    """
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces

### Load Models
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
    image_size=160, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### Load Encoded Features from Dataset
saved_pictures = "UAS/data"  # Folder containing saved images
all_people_faces = {}  # Dictionary to store embeddings of known faces

for file in os.listdir(saved_pictures):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        person_face = os.path.splitext(file)[0]
        img_path = os.path.join(saved_pictures, file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face and extract embeddings
        batch_boxes, faces = mtcnn.detect_box(img_rgb)
        if faces is not None and len(faces) > 0:
            face_tensor = faces[0].unsqueeze(0)  # Take the first detected face
            all_people_faces[person_face] = encode(face_tensor)[0]

### Real-Time Face Detection
def detect(cam=0, thres=0.8):
    """
    Perform real-time face recognition using the webcam.
    """
    vdo = cv2.VideoCapture(cam)
    while vdo.isOpened():
        _, img0 = vdo.read()
        if img0 is None:
            break

        # Detect faces in the current frame
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        batch_boxes, cropped_images = mtcnn.detect_box(img_rgb)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(coord) for coord in box]
                img_embedding = encode(cropped.unsqueeze(0))
                
                # Compare with known faces
                detect_dict = {}
                for k, v in all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                
                # Find the closest match
                min_key = min(detect_dict, key=detect_dict.get)
                if detect_dict[min_key] >= thres:
                    min_key = 'Unknown'

                # Draw bounding box and label
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img0, min_key, (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )
        
        # Display the output
        cv2.imshow("Face Recognition", img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    vdo.release()
    cv2.destroyAllWindows()

### Main Entry Point
if __name__ == "__main__":
    detect(0)