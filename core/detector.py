import os
import logging
from typing import Union, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    # We will raise a clean import error if the environment is not set up
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SkySentry.Core.Detector")


class YOLO11Detector:
    """
    YOLO11 Detector wrapper class for running inference using YOLOv11n from ultralytics.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        """
        Initializes the YOLO11 detector.

        Args:
            model_name: The name or path of the pretrained model (e.g. 'yolo11n.pt').
            device: Computing device ('cpu', 'cuda', 'mps', or 'auto').
            conf: Confidence threshold for filtering detections.
            iou: Intersection over union (IoU) threshold for non-maximum suppression (NMS).
        """
        # Set configuration parameters
        self.model_name = model_name
        self.conf = conf
        self.iou = iou

        # Select the target device automatically if 'auto' is chosen
        if device == "auto":
            self.device = self._select_device()
        else:
            self.device = device

        logger.info(f"Initializing {self.model_name} on device: {self.device}")

        # Lazy load model during first usage or explicitly on init
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info("YOLO model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise e

    def _select_device(self) -> str:
        """
        Automatically selects the best available device for acceleration.
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def predict(
        self,
        source: Union[str, np.ndarray, Image.Image],
        conf: float = None,
        iou: float = None,
        imgsz: int = 640,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Performs inference on the provided source.

        Args:
            source: Source image for detection (filepath, numpy array, or PIL Image).
            conf: Override confidence threshold.
            iou: Override IoU threshold.
            imgsz: Inference image size.
            **kwargs: Additional parameters passed to the model.

        Returns:
            A list of dictionary detections, where each dict has:
            - 'box': [xmin, ymin, xmax, ymax]
            - 'confidence': detection confidence score
            - 'class_id': integer ID of the detected class
            - 'class_name': name of the detected class
        """
        conf_val = conf if conf is not None else self.conf
        iou_val = iou if iou is not None else self.iou

        # Perform prediction
        results = self.model.predict(
            source=source,
            conf=conf_val,
            iou=iou_val,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            **kwargs
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Convert tensor coordinates to list of floats
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                detections.append({
                    "box": xyxy,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })

        return detections

    def predict_and_plot(
        self,
        source: Union[str, np.ndarray, Image.Image],
        conf: float = None,
        iou: float = None,
        imgsz: int = 640,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Performs inference and draws the detected bounding boxes on the image.

        Args:
            source: Source image for detection (filepath, numpy array, or PIL Image).
            conf: Override confidence threshold.
            iou: Override IoU threshold.
            imgsz: Inference image size.
            **kwargs: Additional parameters passed to the model.

        Returns:
            A tuple of:
            - List of detection dictionaries
            - Numpy array of the plotted image (BGR format, suitable for OpenCV)
        """
        conf_val = conf if conf is not None else self.conf
        iou_val = iou if iou is not None else self.iou

        results = self.model.predict(
            source=source,
            conf=conf_val,
            iou=iou_val,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            **kwargs
        )

        detections = []
        # Get the plotted image (returns numpy array in BGR format)
        plotted_img = None

        for result in results:
            if plotted_img is None:
                plotted_img = result.plot()  # Plotted output image

            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                detections.append({
                    "box": xyxy,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })

        # Fallback if no result plotting was done (e.g. empty results)
        if plotted_img is None:
            if isinstance(source, str):
                plotted_img = np.array(Image.open(source))
            elif isinstance(source, Image.Image):
                plotted_img = np.array(source)
            else:
                plotted_img = source

        return detections, plotted_img


if __name__ == "__main__":
    # Test script to verify the installation and execution
    import sys
    print("Testing YOLO11Detector wrapper setup...")
    try:
        detector = YOLO11Detector()
        print("Successfully initialized YOLO11Detector!")
        
        # Test prediction on a dummy blank image if desired or a standard test url
        # Using a solid black 640x640 image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        print("Running prediction on dummy image...")
        detections = detector.predict(dummy_img)
        print(f"Prediction successful! Found {len(detections)} objects.")
    except Exception as e:
        print(f"Error during test: {e}")
        sys.exit(1)
