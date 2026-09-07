"""YOLOv8 Part Detector Module.

This module provides YOLOv8-based object detection for auto parts.
"""
from ultralytics import YOLO
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of part detection."""
    part_type: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    class_name: str = ""


class YOLOPartDetector:
    """YOLOv8-based auto part detector."""
    
    # Auto parts categories mapping
    # Maps YOLO class names to auto part categories
    AUTO_PARTS_CATEGORIES = {
        # Body / Exterior (classes 0-12 + 58, 60, 62)
        'bumper': ['front_bumper', 'rear_bumper'],
        'hood': ['hood'],
        'trunk': ['trunk'],
        'door': ['front_door', 'rear_door'],
        'lighting': ['front_light', 'rear_light', 'headlights', 'taillights'],
        'glass': ['front_windshield', 'rear_windshield', 'window_regulator'],
        'wheel': ['wheel', 'rim'],
        'mirror': ['mirror', 'side_mirror'],
        'grille': ['grille'],
        'spoiler': ['spoiler'],
        # Braking (classes 36, 46, 48, 52)
        'brake': ['brake_pad', 'brake_rotor', 'brake_caliper', 'vacuum_brake_booster'],
        # Filters (classes 23, 67)
        'filter': ['oil_filter', 'air_filter', 'gasoline_filter'],
        # Battery / Electrical (classes 17, 37, 39, 44, 55, 56, 57, 59)
        'battery': ['battery'],
        'alternator': ['alternator', 'starter', 'ignition_coil', 'spark_plug',
                       'distributor', 'oxygen_sensor', 'oil_pressure_sensor',
                       'fuse_box', 'air_compressor'],
        # Engine / Powertrain (classes 18, 25, 27, 29, 30, 35, 40, 43, 45, 47, 49, 50, 53, 54, 68, 71, 72)
        'engine': ['piston', 'engine_block', 'cylinder_head', 'camshaft',
                   'crankshaft', 'engine_valve', 'valve_lifter',
                   'fuel_injector', 'carberator', 'gas_cap',
                   'transmission', 'torque_converter', 'clutch_plate',
                   'pressure_plate', 'oil_pan', 'bearing', 'chain', 'clutch'],
        # Exhaust (class 31)
        'exhaust': ['muffler'],
        # Suspension / Steering (classes 32, 41, 42, 51)
        'suspension': ['lower_control_arm', 'idler_arm', 'leaf_spring', 'coil_spring'],
        # Cooling (classes 14, 20, 21, 22, 28)
        'radiator': ['radiator', 'radiator_hose', 'radiator_fan',
                     'overflow_tank', 'water_pump', 'thermostat'],
        # Interior (classes 19, 33, 61)
        'interior': ['radio', 'instrument_cluster', 'shift_knob'],
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False,
        min_area_pct: float = 2.0  # Ignore detections smaller than 2% of the image
    ):
        """Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            use_gpu: Whether to use GPU for inference
            min_area_pct: Minimum area as percentage of total image (0-100)
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.min_area_pct = min_area_pct
        self._model = None
        
        logger.info(f"Initializing YOLO detector with model: {model_path} (min_area: {min_area_pct}%)")
    
    @property
    def model(self) -> YOLO:
        """Lazy-load the model."""
        if self._model is None:
            logger.info(f"Loading YOLO model from {self.model_path}...")
            self._model = YOLO(self.model_path)
        return self._model
    
    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        """Detect auto part in image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            DetectionResult if part detected and passes size check, None otherwise
        """
        try:
            results = self.model(image, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                logger.debug("No objects detected in image")
                return None
            
            # Get detections that meet confidence threshold
            boxes = results[0].boxes
            valid_detections = []
            
            img_h, img_w = image.shape[:2]
            total_area = img_w * img_h
            
            for i in range(len(boxes)):
                confidence = boxes.conf[i].item()
                if confidence < self.confidence_threshold:
                    continue
                
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = bbox
                
                # Calculate area percentage
                box_area = (x2 - x1) * (y2 - y1)
                area_pct = (box_area / total_area) * 100
                
                if area_pct < self.min_area_pct:
                    logger.debug(f"Ignoring detection: area {area_pct:.2f}% < {self.min_area_pct}% threshold")
                    continue
                
                valid_detections.append({
                    'idx': i,
                    'confidence': confidence,
                    'area_pct': area_pct,
                    'bbox': bbox,
                    'cls_id': int(boxes.cls[i].item())
                })
            
            if not valid_detections:
                logger.debug("No detections met both confidence and size thresholds")
                return None
            
            # Return the highest confidence detection that passed size check
            best = max(valid_detections, key=lambda x: x['confidence'])
            
            class_name = self.model.names[best['cls_id']]
            part_type = self._map_to_part_type(class_name)
            
            logger.info(f"Detected: {class_name} -> {part_type} (conf: {best['confidence']:.2f}, area: {best['area_pct']:.1f}%)")
            
            return DetectionResult(
                part_type=part_type,
                confidence=best['confidence'],
                bbox=best['bbox'],
                class_name=class_name
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None
    
    def detect_all(self, image: np.ndarray, top_k: int = 5) -> List[DetectionResult]:
        """Detect all auto parts in image.
        
        Args:
            image: Input image as numpy array (RGB)
            top_k: Maximum number of detections to return
            
        Returns:
            List of DetectionResult objects that pass size check
        """
        try:
            results = self.model(image, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                return []
            
            boxes = results[0].boxes
            img_h, img_w = image.shape[:2]
            total_area = img_w * img_h
            
            # Filter by confidence and size, then sort
            detections = []
            for i in range(len(boxes)):
                confidence = boxes.conf[i].item()
                if confidence < self.confidence_threshold:
                    continue
                
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = bbox
                
                # Area check
                box_area = (x2 - x1) * (y2 - y1)
                area_pct = (box_area / total_area) * 100
                
                if area_pct < self.min_area_pct:
                    continue
                    
                cls_id = int(boxes.cls[i].item())
                class_name = self.model.names[cls_id]
                part_type = self._map_to_part_type(class_name)
                
                detections.append(DetectionResult(
                    part_type=part_type,
                    confidence=confidence,
                    bbox=bbox,
                    class_name=class_name
                ))
            
            # Sort by confidence and return top_k
            detections.sort(key=lambda x: x.confidence, reverse=True)
            return detections[:top_k]
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _map_to_part_type(self, class_name: str) -> str:
        """Map YOLO class name to auto part category.
        
        Args:
            class_name: YOLO detected class name
            
        Returns:
            Auto part category string
        """
        class_lower = class_name.lower()
        
        for category, keywords in self.AUTO_PARTS_CATEGORIES.items():
            if any(kw in class_lower for kw in keywords):
                return category
        
        # If no mapping found, check for car-related objects
        car_related = ['car', 'vehicle', 'truck', 'motorcycle', 'wheel', 'tire']
        if any(kw in class_lower for kw in car_related):
            return "automotive"
        
        return "unknown"
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported auto part categories.
        
        Returns:
            List of category names
        """
        return list(self.AUTO_PARTS_CATEGORIES.keys())
