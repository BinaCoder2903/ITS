from ultralytics import YOLO
import cv2
import os
import csv
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 1.mp4")
DISPLAY_SCALE = 0.6

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)
VIOLATION_CSV = os.path.join(OUTPUT_DIR, "violations_log.csv")

# ROI phạt (giảm false positive):
# Chỉ tính vi phạm khi tâm bbox nằm dưới ENFORCE_Y_MIN_NORM (0..1)
ENFORCE_Y_MIN_NORM = 0.65  # chỉnh 0.45~0.7 tùy video

# Evidence
SAVE_EVIDENCE = True
SAVE_FULLFRAME = False      # True: lưu cả frame; False: chỉ lưu crop
CROP_PADDING = 0.25         # padding quanh bbox (25%)

# Road polygon (norm 0..1): [left_bottom, right_bottom, right_top, left_top]
ROAD_POLY_NORM = [
    (0.2250, 0.9583),
    (0.7417, 0.9704),
    (0.5620, 0.4481),
    (0.4354, 0.4574),
]

BOUNDARY_TS = [0.0000, 0.3224, 0.6289, 1.0000]


# Lane types (mỗi lane 1 type)
LANE_TYPES = ["motorbike", "car", "car"]  



def lerp(p1, p2, t: float):
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)

def generate_lane_polys_norm(road_poly_norm, boundary_ts, lane_types):
    assert len(road_poly_norm) == 4
    lb, rb, rt, lt = road_poly_norm

    boundary_ts = sorted([max(0.0, min(1.0, float(t))) for t in boundary_ts])
    num_lanes = len(boundary_ts) - 1
    assert num_lanes >= 1

    if len(lane_types) < num_lanes:
        lane_types = lane_types + ["car"] * (num_lanes - len(lane_types))
    lane_types = lane_types[:num_lanes]

    lanes = []
    for i in range(num_lanes):
        tL, tR = boundary_ts[i], boundary_ts[i + 1]
        lane_id = i + 1
        lane_type = lane_types[i]

        bL = lerp(lb, rb, tL)
        bR = lerp(lb, rb, tR)
        tR_pt = lerp(lt, rt, tR)
        tL_pt = lerp(lt, rt, tL)

        lanes.append({
            "id": lane_id,
            "name": f"L{lane_id}",
            "type": lane_type,
            "poly_norm": [bL, bR, tR_pt, tL_pt],
        })
    return lanes

LANES_CONFIG = generate_lane_polys_norm(ROAD_POLY_NORM, BOUNDARY_TS, LANE_TYPES)

def build_allowed_lanes_by_class(lanes_cfg):
    lanes_by_type: Dict[str, List[int]] = {}
    for lane in lanes_cfg:
        lanes_by_type.setdefault(lane.get("type", "unknown"), []).append(lane["id"])

    car_lanes = set(lanes_by_type.get("car", []))
    bike_lanes = set(lanes_by_type.get("motorbike", []))

    return {
        "car": car_lanes,
        "bus": car_lanes,
        "truck": car_lanes,
        "motorbike": bike_lanes,
    }

ALLOWED_LANES_BY_CLASS = build_allowed_lanes_by_class(LANES_CONFIG)

def denorm_polygon(poly_norm, w: int, h: int):
    return [(int(x * w), int(y * h)) for x, y in poly_norm]

def point_in_poly(x: float, y: float, poly: List[Tuple[int, int]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-8) + x1):
            inside = not inside
    return inside

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_with_padding(frame, bbox, pad_ratio=0.25):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * pad_ratio
    pad_h = bh * pad_ratio

    cx1 = int(clamp(x1 - pad_w, 0, w - 1))
    cy1 = int(clamp(y1 - pad_h, 0, h - 1))
    cx2 = int(clamp(x2 + pad_w, 0, w - 1))
    cy2 = int(clamp(y2 + pad_h, 0, h - 1))
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)

def save_evidence(frame, track_id: int, cls_name: str, lane_id: Optional[int], time_sec: float, bbox_int):
    """
    Lưu evidence crop (và optional fullframe) vào output/evidence/
    Trả về path file crop.
    """
    if not SAVE_EVIDENCE:
        return ""

    crop_img, crop_box = crop_with_padding(frame, bbox_int, CROP_PADDING)

    # vẽ bbox lên crop để nhìn rõ
    cx1, cy1, cx2, cy2 = crop_box
    x1, y1, x2, y2 = bbox_int
    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1
    cv2.rectangle(crop_img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
    label = f"ID{track_id} {cls_name} L{lane_id} t={time_sec:.1f}s"
    cv2.putText(crop_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    fname = f"v_{int(time_sec*10):06d}_id{track_id}_{cls_name}_L{lane_id}.jpg"
    crop_path = os.path.join(EVIDENCE_DIR, fname)
    cv2.imwrite(crop_path, crop_img)

    if SAVE_FULLFRAME:
        full_name = f"full_{int(time_sec*10):06d}_id{track_id}.jpg"
        full_path = os.path.join(EVIDENCE_DIR, full_name)
        cv2.imwrite(full_path, frame)

    return crop_path.replace("\\", "/")  # cho CSV đọc dễ


# ============================================================
#  TRACKER ĐƠN GIẢN (IOU)
# ============================================================

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    cls_name: str
    conf: float

@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    cls_name: str
    conf: float
    missed: int = 0
    lane_id: Optional[int] = None
    lane_type: Optional[str] = None
    is_violation: bool = False

def iou(box1, box2) -> float:
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    a1 = (x2 - x1) * (y2 - y1)
    a2 = (x2g - x1g) * (y2g - y1g)
    return inter / (a1 + a2 - inter + 1e-8)

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missed=10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        if not self.tracks:
            for det in detections:
                self._create(det)
            return list(self.tracks.values())

        unmatched = list(range(len(detections)))
        track_ids = list(self.tracks.keys())
        matches: Dict[int, int] = {}

        for tid in track_ids:
            best_iou = 0.0
            best_idx = -1
            for di in unmatched:
                v = iou(self.tracks[tid].bbox, detections[di].bbox)
                if v > best_iou:
                    best_iou = v
                    best_idx = di
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                matches[tid] = best_idx
                unmatched.remove(best_idx)

        for tid, di in matches.items():
            det = detections[di]
            tr = self.tracks[tid]
            tr.bbox = det.bbox
            tr.cls_name = det.cls_name
            tr.conf = det.conf
            tr.missed = 0

        for tid in track_ids:
            if tid not in matches:
                self.tracks[tid].missed += 1

        for tid in [tid for tid, tr in self.tracks.items() if tr.missed > self.max_missed]:
            del self.tracks[tid]

        for di in unmatched:
            self._create(detections[di])

        return list(self.tracks.values())

    def _create(self, det: Detection):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = Track(tid, det.bbox, det.cls_name, det.conf)


# ============================================================
#  LANE ASSIGN + VIOLATION
# ============================================================

def assign_lane(track: Track, lanes_pixel: List[dict]) -> Optional[int]:
    x1, y1, x2, y2 = track.bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    for lane in lanes_pixel:
        if point_in_poly(cx, cy, lane["poly"]):
            return lane["id"]
    return None

def is_in_enforce_zone(track: Track, frame_h: int) -> bool:
    x1, y1, x2, y2 = track.bbox
    cy = (y1 + y2) / 2.0
    return cy >= ENFORCE_Y_MIN_NORM * frame_h

def is_violation(track: Track) -> bool:
    allowed = ALLOWED_LANES_BY_CLASS.get(track.cls_name)
    if not allowed or track.lane_id is None:
        return False
    return track.lane_id not in allowed


# ============================================================
#  MAIN
# ============================================================

def main():
    print("[INFO] VIDEO_SOURCE:", VIDEO_SOURCE)
    print("[INFO] CSV output:", VIOLATION_CSV)

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Không mở được video:", VIDEO_SOURCE)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    tracker = SimpleTracker(iou_threshold=0.3, max_missed=10)
    target_classes = {"car", "motorbike", "bus", "truck"}

    # track_id đã log vi phạm (unique)
    logged_violation_ids = set()

    with open(VIOLATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "time_sec", "track_id", "class", "conf",
            "lane_id", "lane_type", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "evidence_path"
        ])

        cv2.namedWindow("Lane Violation Demo", cv2.WINDOW_NORMAL)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            time_sec = frame_idx / fps
            h, w = frame.shape[:2]

            # Build lanes pixel
            lanes_pixel = []
            lane_type_by_id = {}
            for lane_cfg in LANES_CONFIG:
                pts = denorm_polygon(lane_cfg["poly_norm"], w, h)
                lanes_pixel.append({
                    "id": lane_cfg["id"],
                    "name": lane_cfg["name"],
                    "type": lane_cfg["type"],
                    "poly": pts,
                })
                lane_type_by_id[lane_cfg["id"]] = lane_cfg["type"]

            # Draw road polygon
            road_pts = denorm_polygon(ROAD_POLY_NORM, w, h)
            cv2.polylines(frame, [np.array(road_pts, np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2)

            # Draw enforce zone line (optional)
            y_line = int(ENFORCE_Y_MIN_NORM * h)
            cv2.line(frame, (0, y_line), (w, y_line), (80, 80, 80), 2)

            # YOLO detect
            results = model(frame, conf=0.4)[0]
            detections: List[Detection] = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name not in target_classes:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(Detection((x1, y1, x2, y2), cls_name, conf))

            tracks = tracker.update(detections)

            # Counts
            lane_counts = {lane["id"]: 0 for lane in lanes_pixel}
            violations_frame = 0

            # Assign lane + check
            for tr in tracks:
                tr.lane_id = assign_lane(tr, lanes_pixel)
                tr.lane_type = lane_type_by_id.get(tr.lane_id) if tr.lane_id else None

                if tr.lane_id is not None:
                    lane_counts[tr.lane_id] += 1

                # chỉ xét vi phạm trong vùng enforce zone
                tr.is_violation = False
                if tr.lane_id is not None and is_in_enforce_zone(tr, h):
                    tr.is_violation = is_violation(tr)

                if tr.is_violation:
                    violations_frame += 1

                    if tr.track_id not in logged_violation_ids:
                        bbox_int = tuple(map(int, tr.bbox))
                        evidence_path = save_evidence(frame, tr.track_id, tr.cls_name, tr.lane_id, time_sec, bbox_int)

                        writer.writerow([
                            frame_idx, round(time_sec, 2), tr.track_id, tr.cls_name, round(tr.conf, 3),
                            tr.lane_id, tr.lane_type,
                            bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3],
                            evidence_path
                        ])
                        logged_violation_ids.add(tr.track_id)

            # Draw lanes
            for lane in lanes_pixel:
                pts = np.array(lane["poly"], np.int32).reshape(-1, 1, 2)
                t = lane["type"]
                if t == "car":
                    color = (255, 0, 0)
                elif t == "motorbike":
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 255)
                cv2.polylines(frame, [pts], True, color, 2)

                cx = int(sum(p[0] for p in lane["poly"]) / len(lane["poly"]))
                cy = int(sum(p[1] for p in lane["poly"]) / len(lane["poly"]))
                cv2.putText(frame, lane["name"], (cx - 18, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Draw vehicles (text only for violation)
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr.bbox)
                if tr.is_violation:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "WRONG LANE", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Overlay summary
            l1 = lane_counts.get(1, 0)
            l2 = lane_counts.get(2, 0)
            l3 = lane_counts.get(3, 0)

            info1 = f"Frame: {frame_idx}  t={time_sec:5.1f}s"
            info2 = f"L1:{l1:2}  L2:{l2:2}  L3:{l3:2}  Veh:{len(tracks):2}"
            info3 = f"Viol(frame): {violations_frame}  Viol(unique): {len(logged_violation_ids)}"

            cv2.putText(frame, info1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, info2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, info3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Show
            new_w, new_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
            disp = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Lane Violation Demo", disp)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Saved CSV:", VIOLATION_CSV)
    print("[INFO] Evidence dir:", EVIDENCE_DIR)


if __name__ == "__main__":
    main()
