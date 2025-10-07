import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("yolo11m-seg.pt")

# Video input
cap = cv2.VideoCapture(0)

# --- Step 1: Draw ROI polygon manually ---
roi_points = []
roi_finalized = False

def draw_roi(event, x, y, flags, param):
    global roi_points, roi_finalized
    if event == cv2.EVENT_LBUTTONDOWN and not roi_finalized:
        roi_points.append((x, y))

# Grab first frame
ret, frame = cap.read()
if not ret:
    print("❌ Could not read video")
    cap.release()
    exit()

clone = frame.copy()
cv2.namedWindow("Draw ROI")
cv2.setMouseCallback("Draw ROI", draw_roi)

while True:
    temp_frame = clone.copy()

    # Draw clicked points and connecting lines
    if len(roi_points) > 1:
        cv2.polylines(temp_frame, [np.array(roi_points, np.int32)], False, (255, 255, 0), 2)
    for p in roi_points:
        cv2.circle(temp_frame, p, 5, (0, 0, 255), -1)

    cv2.putText(temp_frame, "Click 5+ points, press ENTER to finish",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Draw ROI", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and len(roi_points) >= 3:  # ENTER to finalize
        roi_finalized = True
        break
    elif key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw ROI")
ROI_POLY = np.array(roi_points, np.int32)
print(f"✅ ROI polygon set with {len(roi_points)} points.")

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Step 2: Detection + Segmentation ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, iou=0.4, imgsz=1280, verbose=False)
    people_in_roi = 0

    for r in results:
        if r.masks is not None and r.boxes is not None:
            for mask, cls in zip(r.masks.xy, r.boxes.cls):
                if int(cls) == 0:  # person
                    poly = np.array(mask, dtype=np.int32)

                    # Person centroid
                    M = cv2.moments(poly)
                    if M["m00"] > 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                        # Check if centroid is inside ROI polygon
                        if cv2.pointPolygonTest(ROI_POLY, (cx, cy), False) >= 0:
                            people_in_roi += 1
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [poly], (0, 255, 0))
                            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Draw ROI polygon
    cv2.polylines(frame, [ROI_POLY], True, (255, 255, 0), 2)
    cv2.putText(frame, "ROI", (ROI_POLY[0][0], ROI_POLY[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show people count
    cv2.putText(frame, f"People in ROI: {people_in_roi}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Alert if more than 3 people
    if people_in_roi > 3:
        cv2.putText(frame, "ALERT: Too Many People!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("ROI Segmentation + Alert", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
