import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime
import json

class FixedPeopleTracker:
    def __init__(self):
        # === YOLOv11n CONFIG ===
        self.model = YOLO('yolov11n.pt')
        
        # === CONSERVATIVE PARAMETERS ===
        self.conf_threshold = 0.6        # Higher threshold for better precision
        self.iou_threshold = 0.5         # Standard NMS
        
        # === STRICT VALIDATION ===
        self.min_bbox_area = 1000        # Increased minimum area
        self.max_bbox_area = 100000      # Maximum area
        self.min_aspect_ratio = 0.3      # Minimum width/height
        self.max_aspect_ratio = 3.0      # Maximum width/height
        self.min_height = 40             # Minimum pixel height
        
        # === TRACKING STABILITY ===
        self.min_track_frames = 5        # Minimum frames before counting
        self.track_buffer = 25           # Keep lost tracks
        
        # === RESULTS STORAGE ===
        self.create_results_folder()
        
        # Tracking variables
        self.track_history = {}
        self.track_start_frame = {}
        self.people_count = 0
        self.counted_ids = set()
        self.detection_data = []
        self.frame_count = 0
        
    def create_results_folder(self):
        """Create Revised folder for storing results"""
        self.results_folder = "Revised"
        os.makedirs(self.results_folder, exist_ok=True)
        
        self.video_output_folder = os.path.join(self.results_folder, "videos")
        self.data_output_folder = os.path.join(self.results_folder, "data")
        os.makedirs(self.video_output_folder, exist_ok=True)
        os.makedirs(self.data_output_folder, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Results will be saved in: {self.results_folder}")
        
    def setup_video_writer(self, frame_width, frame_height, fps=30):
        """Setup video writer to save output"""
        video_filename = os.path.join(self.video_output_folder, f"fixed_tracking_{self.timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        print(f"Video will be saved as: {video_filename}")
        
    def validate_detection(self, bbox, confidence, frame_shape):
        """
        Strict validation of detections
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate dimensions
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Strict validation checks
        checks = [
            confidence >= self.conf_threshold,           # Confidence threshold
            area >= self.min_bbox_area,                  # Minimum size
            area <= self.max_bbox_area,                  # Maximum size
            height >= self.min_height,                   # Minimum height
            self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio,  # Aspect ratio
            x1 >= 0 and y1 >= 0,                        # Within frame bounds
            x2 <= frame_shape[1] and y2 <= frame_shape[0], # Within frame bounds
            width > 20,                                 # Minimum width
        ]
        
        is_valid = all(checks)
        
        if not is_valid:
            # Print why it was rejected (for debugging)
            rejection_reasons = []
            if confidence < self.conf_threshold: rejection_reasons.append(f"low confidence ({confidence:.2f})")
            if area < self.min_bbox_area: rejection_reasons.append(f"small area ({area:.0f})")
            if height < self.min_height: rejection_reasons.append(f"short height ({height:.0f})")
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio: 
                rejection_reasons.append(f"bad aspect ratio ({aspect_ratio:.2f})")
            
            if rejection_reasons:
                print(f"Rejected detection: {', '.join(rejection_reasons)}")
        
        return is_valid, area, aspect_ratio
    
    def track_people(self, frame):
        """
        Reliable tracking with error handling
        """
        self.frame_count += 1
        
        try:
            # Use predict instead of track for more control
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],  # Person class only
                verbose=False
            )
            
            return results
            
        except Exception as e:
            print(f"Tracking error: {e}")
            return None
    
    def simple_tracking(self, detections, frame):
        """
        Simple but reliable manual tracking
        """
        current_tracks = {}
        
        for detection in detections:
            bbox = detection['bbox_raw']
            confidence = detection['confidence']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Find closest existing track
            track_id = None
            min_distance = 50  # pixels
            
            for existing_id, history in self.track_history.items():
                if history:  # Check if history exists
                    last_center_x, last_center_y = history[-1][1:3]
                    distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        track_id = existing_id
            
            # Create new track if no close match
            if track_id is None:
                track_id = len(self.track_history) + 1
                self.track_start_frame[track_id] = self.frame_count
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((self.frame_count, center_x, center_y, bbox, confidence))
            
            # Keep only recent history
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
            
            current_tracks[track_id] = {
                'bbox': bbox,
                'confidence': confidence,
                'center': (center_x, center_y)
            }
        
        return current_tracks
    
    def remove_stale_tracks(self):
        """Remove tracks that haven't been updated"""
        stale_tracks = []
        for track_id, history in self.track_history.items():
            if history and self.frame_count - history[-1][0] > self.track_buffer:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_start_frame:
                del self.track_start_frame[track_id]
    
    def update_counting(self, current_tracks):
        """
        Update counting with stability checks
        """
        current_ids = set(current_tracks.keys())
        
        # Only count tracks that have been stable for minimum frames
        valid_new_ids = []
        for track_id in current_ids:
            if (track_id not in self.counted_ids and 
                track_id in self.track_start_frame and
                self.frame_count - self.track_start_frame[track_id] >= self.min_track_frames):
                valid_new_ids.append(track_id)
        
        if valid_new_ids:
            self.people_count += len(valid_new_ids)
            self.counted_ids.update(valid_new_ids)
            print(f"✅ Counted new people: {valid_new_ids} (Total: {self.people_count})")
    
    def process_detections(self, results, frame):
        """
        Process and validate detections
        """
        valid_detections = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                for i, (bbox, confidence) in enumerate(zip(boxes_xyxy, confidences)):
                    x1, y1, x2, y2 = bbox
                    
                    # Validate detection
                    is_valid, area, aspect_ratio = self.validate_detection(bbox, confidence, frame.shape)
                    
                    if is_valid:
                        valid_detections.append({
                            'bbox_raw': bbox,
                            'confidence': confidence,
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
        
        return valid_detections
    
    def visualize_clean_results(self, frame, tracks):
        """
        Clean and professional visualization
        """
        # Remove stale tracks first
        self.remove_stale_tracks()
        
        # Draw current tracks
        for track_id, track_info in tracks.items():
            bbox = track_info['bbox']
            confidence = track_info['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate age of track
            track_age = self.frame_count - self.track_start_frame.get(track_id, self.frame_count)
            
            # Color based on track stability
            if track_age > 30:
                color = (0, 255, 0)  # Green - stable track
            elif track_age > 10:
                color = (0, 255, 255)  # Yellow - medium stability
            else:
                color = (0, 0, 255)  # Red - new track
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw info with clean formatting
            info_text = f"ID:{track_id} C:{confidence:.2f}"
            cv2.rectangle(frame, (x1, y1-25), (x1 + 120, y1), color, -1)
            cv2.putText(frame, info_text, (x1+2, y1-8),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw center point
            center_x, center_y = track_info['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 3, color, -1)
            
            # Draw track history
            if track_id in self.track_history:
                points = []
                for _, cx, cy, _, _ in self.track_history[track_id][-20:]:  # Last 20 points
                    points.append([int(cx), int(cy)])
                
                if len(points) > 1:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 1)
        
        # Update counting
        self.update_counting(tracks)
        
        # Display clean statistics
        stats = [
            f"People in frame: {len(tracks)}",
            f"Total counted: {self.people_count",
            f"Active tracks: {len(self.track_history)}",
            f"Frame: {self.frame_count}"
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (10, 30 + i*25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def save_clean_data(self, tracks):
        """Save clean detection data"""
        frame_data = {
            "frame": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "people_in_frame": len(tracks),
            "total_counted": self.people_count,
            "tracks": []
        }
        
        for track_id, track_info in tracks.items():
            track_data = {
                "id": track_id,
                "confidence": float(track_info['confidence']),
                "bbox": [float(x) for x in track_info['bbox']],
                "age": self.frame_count - self.track_start_frame.get(track_id, self.frame_count)
            }
            frame_data["tracks"].append(track_data)
        
        self.detection_data.append(frame_data)
    
    def export_results(self):
        """Export clean results"""
        json_filename = os.path.join(self.data_output_folder, f"clean_tracking_{self.timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(self.detection_data, f, indent=2)
        
        summary_filename = os.path.join(self.data_output_folder, f"summary_{self.timestamp}.txt")
        with open(summary_filename, 'w') as f:
            f.write("=== CLEAN TRACKING RESULTS ===\n")
            f.write(f"Total Frames: {self.frame_count}\n")
            f.write(f"Total People Counted: {self.people_count}\n")
            f.write(f"Final Active Tracks: {len(self.track_history)}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write(f"Min Track Frames: {self.min_track_frames}\n")
        
        print(f"✅ Clean results exported to: {self.results_folder}")

def main():
    # Initialize tracker
    tracker = FixedPeopleTracker()
    
    print("=== CLEAN YOLOv11n TRACKING ===")
    print("Using conservative parameters for reliable results")
    print("Press 'q' to quit, 'r' to reset, 'c' to clear tracks")
    
    # Video source
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    tracker.setup_video_writer(frame_width, frame_height)
    
    print("🚀 Starting clean tracking...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = tracker.track_people(frame)
            
            if results:
                # Process detections with strict validation
                detections = tracker.process_detections(results, frame)
                
                # Apply simple but reliable tracking
                tracks = tracker.simple_tracking(detections, frame)
                
                # Visualize with clean output
                frame = tracker.visualize_clean_results(frame, tracks)
                
                # Save clean data
                tracker.save_clean_data(tracks)
                
                # Print clean progress
                if tracker.frame_count % 30 == 0:
                    print(f"Frame {tracker.frame_count}: {len(tracks)} people, {len(tracker.track_history)} tracks")
            
            # Write to video
            tracker.video_writer.write(frame)
            
            # Show frame
            cv2.imshow('Clean People Tracking', frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.people_count = 0
                tracker.counted_ids.clear()
                print("🔄 Counting reset!")
            elif key == ord('c'):
                tracker.track_history.clear()
                tracker.track_start_frame.clear()
                print("🗑️  Tracks cleared!")
                
    except KeyboardInterrupt:
        print("⏹️ Stopped by user")
    
    # Cleanup
    cap.release()
    tracker.video_writer.release()
    tracker.export_results()
    cv2.destroyAllWindows()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total people counted: {tracker.people_count}")
    print(f"Frames processed: {tracker.frame_count}")

if __name__ == "__main__":
    main()
