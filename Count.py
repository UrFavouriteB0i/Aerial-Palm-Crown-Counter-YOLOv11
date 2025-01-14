import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import os
import matplotlib.pyplot as plt

model_weight_path = 'weights/best.pt'
model = YOLO(model_weight_path)

def infer_img(image_path):
    #Read the image
    image = cv2.imread(image_path)

    #Performance inference to predict object
    # from the input image
    result = model.predict(image)[0]

    #Converting YOLO prediction result into supervision format
    detection = sv.Detections.from_ultralytics(result)
    
    #Build custom box and label annotator to meet criteria

    #Determine the annotator parameter dynamically
    #according to amount of the detected object in the image
    num_objects = len(detection.xyxy)
    border_thickness = 15 if num_objects > 100 else 10
    text_size = 4 if num_objects > 100 else 2
    txt_thickness = 3
    colorbt = sv.Color(255,0,0)

    #initiate the box and label annotator
    box_annotator = sv.BoxAnnotator(thickness=border_thickness, color=colorbt)
    lbl_annotator = sv.LabelAnnotator(text_scale=text_size, text_thickness=txt_thickness)

    #Custom the label format to show numbers of detected palm tree sequentially
    label = [
        f"#{i+1}"
        for i in range(len(detection.xyxy))
    ]

    #assign the custom annotation into the image
    #bounding box annotation setup
    annotated_img = box_annotator.annotate(
        scene=image, detections=detection
    )
    #labels annotation setup
    annotated_img = lbl_annotator.annotate(
        scene=annotated_img, detections=detection, labels=label
    )


    height, width = annotated_img.shape[:2]

    # Set a target width (e.g., 800 pixels) while maintaining the aspect ratio
    target_width = 1080
    aspect_ratio = target_width / float(width)
    target_height = int(height * aspect_ratio)

    # Resize the image to the target size
    resized_image = cv2.resize(annotated_img, (target_width, target_height))

    cv2.imshow('Palm Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'Path/to/Your/images.jpg' # change this into your own image path in local workspace

     # Check if the image file exists
    if os.path.exists(image_path):
        infer_img(image_path)
    else:
        print(f"Image file not found: {image_path}")
    