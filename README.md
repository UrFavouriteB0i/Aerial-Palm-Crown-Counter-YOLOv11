# Aerial Crown Detection Inference App

This repository contains an aerial crown detection inference application built on YOLO (You Only Look Once) v11. The app processes aerial images to detect tree crowns and can be customized with your own image inputs.

## Features

- **Object Detection:** Utilizes YOLO for detecting tree crowns in aerial images.
- **Customization:** Easily change the image input by modifying the `image_path` in the main function.

## Installation

### Prerequisites

- Python 3.10 or later
- Pip (Python package manager)

### Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/UrFavouriteB0i/Aerial-Palm-Crown-Counter-YOLOv11.git
   cd aerial-crown-detection
   ```
### Install the required packages:

- `pip install -r requirements.txt`
- Change the `image_path` in the main function of the script to point to your input image.

## Usage
Run the app with the following command:

```bash
python main.py
```
The app will load the YOLO model, process the specified image, and display or save the detection results.

## Notes
Ensure your input images are in a compatible format (e.g., .jpg, .png).
The YOLO model and its weights should be properly set up before running the inference.

## Source
- [YOLO from ultralytics](https://docs.ultralytics.com/models/yolo11/#what-are-the-key-improvements-in-ultralytics-yolo11-compared-to-previous-versions)
- [Supervision from Roboflow](https://supervision.roboflow.com/latest/)
- [Aerial Palm Crown Dataset for training](https://universe.roboflow.com/mahir-sehmi-fgblg/oil-palm-tree-crown-detection-from-aerial-image/dataset/16)
