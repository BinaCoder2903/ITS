from ultralytics import YOLO
import cv2
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 2.mp4")
DISPLAY_SCALE = 0.6  # thu nhỏ terminal 

print("[DEBUG] VIDEO_SOURCE =", VIDEO_SOURCE)
print("[DEBUG] File video tồn tại? :", os.path.exists(VIDEO_SOURCE))


# lấy frame 
ROAD_POLY_NORM = [
    (0.0719, 0.9454),
    (0.5984, 0.9704),
    (0.5927, 0.5370),
    (0.4484, 0.5315),
]
# lấy lane chia 1 2 3
BOUNDARY_TS = [
    0.0000,
    0.2683,
    0.6335,
    1.0000,
]

# Kiểu lane tương ứng từng khoảng [ti, ti+1]
LANE_TYPES = ["motorbike", "car", "car"]  


def lerp(p1, p2, t: float):
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)


def generate_lane_polys_norm(
    road_poly_norm: List[Tuple[float, float]],
    boundary_ts: List[float],
    lane_types: List[str],
):
    
    assert len(road_poly_norm) == 4, "ROAD_POLY_NORM phải có 4 điểm."
    lb, rb, rt, lt = road_poly_norm

    boundary_ts = [max(0.0, min(1.0, float(t))) for t in boundary_ts]
    boundary_ts = sorted(boundary_ts)
    num_lanes = len(boundary_ts) - 1
    assert num_lanes >= 1, "Cần ít nhất 1 lane."

    if len(lane_types) < num_lanes:
        lane_types = lane_types + ["car"] * (num_lanes - len(lane_types))
    lane_types = lane_types[:num_lanes]

    lanes = []
    for i in range(num_lanes):
        t_left = boundary_ts[i]
        t_right = boundary_ts[i + 1]
        lane_id = i + 1
        lane_type = lane_types[i]

        b_left = lerp(lb, rb, t_left)
        b_right = lerp(lb, rb, t_right)
        t_right_pt = lerp(lt, rt, t_right)
        t_left_pt = lerp(lt, rt, t_left)

        poly_norm = [b_left, b_right, t_right_pt, t_left_pt]

        lanes.append(
            {
                "id": lane_id,
                "name": f"L{lane_id}",  
                "type": lane_type,
                "poly_norm": poly_norm,
            }
        )

    return lanes


LANES_CONFIG = generate_lane_polys_norm(
    ROAD_POLY_NORM, BOUNDARY_TS, LANE_TYPES
)


def build_allowed_lanes_by_class(lanes_cfg) -> Dict[str, set]:
    lanes_by_type: Dict[str, List[int]] = {}
    for lane in lanes_cfg:
        t = lane.get("type", "unknown")
        lanes_by_type.setdefault(t, []).append(lane["id"])

    car_lanes = set(lanes_by_type.get("car", []))
    bike_lanes = set(lanes_by_type.get("motorbike", []))

    return {
        "car": car_lanes,
        "bus": car_lanes,
        "truck": car_lanes,
        "motorbike": bike_lanes,
    }


ALLOWED_LANES_BY_CLASS = build_allowed_lanes_by_class(LANES_CONFIG)


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
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union = box1_area + box2_area - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())

        unmatched_dets = list(range(len(detections)))
        track_ids = list(self.tracks.keys())
        matches: Dict[int, int] = {}

        for tid in track_ids:
            best_iou = 0.0
            best_idx = -1
            for di in unmatched_dets:
                i = iou(self.tracks[tid].bbox, detections[di].bbox)
                if i > best_iou:
                    best_iou = i
                    best_idx = di
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                matches[tid] = best_idx
                unmatched_dets.remove(best_idx)

        for tid, det_idx in matches.items():
            det = detections[det_idx]
            tr = self.tracks[tid]
            tr.bbox = det.bbox
            tr.cls_name = det.cls_name
            tr.conf = det.conf
            tr.missed = 0

        for tid in track_ids:
            if tid not in matches:
                self.tracks[tid].missed += 1

        to_delete = [tid for tid, tr in self.tracks.items()
                     if tr.missed > self.max_missed]
        for tid in to_delete:
            del self.tracks[tid]

        for di in unmatched_dets:
            self._create_track(detections[di])

        return list(self.tracks.values())

    def _create_track(self, det: Detection):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = Track(
            track_id=tid,
            bbox=det.bbox,
            cls_name=det.cls_name,
            conf=det.conf,
        )

#làn xe 

def denorm_polygon(poly_norm, w: int, h: int):
    return [(int(x * w), int(y * h)) for x, y in poly_norm]


def point_in_poly(x: float, y: float, poly: List[Tuple[int, int]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and \
           (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-8) + x1):
            inside = not inside
    return inside


def assign_lane_to_track(track: Track, lanes_pixel: List[dict]) -> Optional[int]:
    x1, y1, x2, y2 = track.bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    for lane in lanes_pixel:
        if point_in_poly(cx, cy, lane["poly"]):
            return lane["id"]
    return None


def check_lane_violation(track: Track) -> bool:
    allowed = ALLOWED_LANES_BY_CLASS.get(track.cls_name)
    if not allowed:
        return False
    if track.lane_id is None:
        return False
    return track.lane_id not in allowed




def main():
    print("[INFO] Đang load YOLOv8...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {VIDEO_SOURCE}")
        return

    target_classes = {"car", "motorbike", "bus", "truck"}
    tracker = SimpleTracker(iou_threshold=0.3, max_missed=10)

    cv2.namedWindow("YOLOv8 Lane RoadPoly Demo", cv2.WINDOW_NORMAL)
    print("[INFO] Bắt đầu. Nhấn 'q' để thoát.")

    frame_idx = 0
    total_violations = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Hết video.")
            break
        frame_idx += 1

        h, w = frame.shape[:2]

        # Lane polygon pixel
        lanes_pixel = []
        lane_type_by_id: Dict[int, str] = {}
        for lane_cfg in LANES_CONFIG:
            pts = denorm_polygon(lane_cfg["poly_norm"], w, h)
            lane = {
                "id": lane_cfg["id"],
                "name": lane_cfg["name"],
                "type": lane_cfg.get("type", "unknown"),
                "poly": pts,
            }
            lanes_pixel.append(lane)
            lane_type_by_id[lane["id"]] = lane["type"]

        # Road polygon (viền ngoài)
        road_pts = denorm_polygon(ROAD_POLY_NORM, w, h)
        cv2.polylines(
            frame,
            [np.array(road_pts, dtype=np.int32).reshape(-1, 1, 2)],
            True,
            (255, 255, 0),
            2,
        )

        # YOLO detection
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

        # Tracking
        tracks = tracker.update(detections)

        # Gán lane + check violation
        violations_frame = 0
        for tr in tracks:
            tr.lane_id = assign_lane_to_track(tr, lanes_pixel)
            tr.lane_type = lane_type_by_id.get(tr.lane_id) if tr.lane_id else None
            tr.is_violation = check_lane_violation(tr)
            if tr.is_violation:
                violations_frame += 1
        total_violations += violations_frame

        # Vẽ lane (hình thang + label L1/L2/L3)
        for lane in lanes_pixel:
            pts = np.array(lane["poly"], dtype=np.int32).reshape(-1, 1, 2)

            lane_type = lane["type"]
            if lane_type == "car":
                color = (255, 0, 0)
            elif lane_type == "motorbike":
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

            cv2.polylines(frame, [pts], True, color, 2)

            cx = int(sum(p[0] for p in lane["poly"]) / len(lane["poly"]))
            cy = int(sum(p[1] for p in lane["poly"]) / len(lane["poly"]))
            cv2.putText(
                frame,
                lane["name"],
                (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        # Vẽ xe: chỉ hiển thị text khi VIOLATION
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.bbox)
            if tr.is_violation:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    "WRONG LANE",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                # chỉ khung xanh, không text để đỡ rối
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Info góc trên trái
        info1 = f"Frame: {frame_idx}"
        info2 = f"Vehicles: {len(tracks)}   Violations(frame): {violations_frame}"
        cv2.putText(
            frame,
            info1,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            info2,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Thu nhỏ hiển thị
        new_w, new_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
        frame_display = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("YOLOv8 Lane RoadPoly Demo", frame_display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát.")
    print("[INFO] Tổng số vi phạm (đếm theo frame):", total_violations)


if __name__ == "__main__":
    main()
