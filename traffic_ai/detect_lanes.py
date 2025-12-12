from ultralytics import YOLO
import cv2
import os
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# ==========================
#  CẤU HÌNH CƠ BẢN
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 1.mp4")
LANE_CONFIG_JSON = os.path.join(BASE_DIR, "auto_lanes_config.json")
DISPLAY_SCALE = 0.6  # thu nhỏ khi hiển thị

print("[DEBUG] VIDEO_SOURCE =", VIDEO_SOURCE)
print("[DEBUG] File video tồn tại? :", os.path.exists(VIDEO_SOURCE))
print("[DEBUG] LANE_CONFIG_JSON =", LANE_CONFIG_JSON)
print("[DEBUG] File lane config tồn tại? :", os.path.exists(LANE_CONFIG_JSON))


# ==========================
#  ĐỌC CONFIG LÀN TỪ JSON
# ==========================

def load_lane_config() -> List[dict]:
    """
    Đọc cấu hình làn từ auto_lanes_config.json nếu có.
    Nếu không có thì dùng config mẫu (hard-code).
    """
    if os.path.exists(LANE_CONFIG_JSON):
        with open(LANE_CONFIG_JSON, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        lanes = []
        for lane in cfg.get("lanes", []):
            poly_norm = [(float(x), float(y)) for x, y in lane["poly_norm"]]
            lane_type = lane.get("type", "unknown")
            lane_id = int(lane["id"])
            lanes.append(
                {
                    "id": lane_id,
                    "name": f"Lane {lane_id} ({lane_type})",
                    "type": lane_type,
                    "poly_norm": poly_norm,
                }
            )
        print("[INFO] Loaded lane config from JSON.")
        return lanes

    # fallback: cấu hình mẫu nếu chưa có JSON
    print("[WARN] Lane config JSON not found. Using default lanes.")
    return [
        {
            "id": 1,
            "name": "Lane 1 (car)",
            "type": "car",
            "poly_norm": [
                (0.15, 0.55),
                (0.45, 0.55),
                (0.75, 1.0),
                (0.02, 1.0),
            ],
        },
        {
            "id": 2,
            "name": "Lane 2 (motorbike)",
            "type": "motorbike",
            "poly_norm": [
                (0.45, 0.55),
                (0.80, 0.55),
                (1.05, 1.0),
                (0.75, 1.0),
            ],
        },
    ]


LANES_CONFIG = load_lane_config()


def build_allowed_lanes_by_class(lanes_cfg: List[dict]) -> Dict[str, set]:
    """
    Từ list lane (có type), xây map:
      - car/bus/truck  -> các lane type = "car"
      - motorbike      -> các lane type = "motorbike"
    """
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


# ==========================
#  TRACKER ĐƠN GIẢN
# ==========================

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

        unmatched_detections = list(range(len(detections)))
        track_ids = list(self.tracks.keys())
        matches: Dict[int, int] = {}

        for tid in track_ids:
            best_iou = 0.0
            best_det_idx = -1
            for di in unmatched_detections:
                i = iou(self.tracks[tid].bbox, detections[di].bbox)
                if i > best_iou:
                    best_iou = i
                    best_det_idx = di
            if best_det_idx >= 0 and best_iou >= self.iou_threshold:
                matches[tid] = best_det_idx
                unmatched_detections.remove(best_det_idx)

        # Cập nhật track match
        for tid, det_idx in matches.items():
            det = detections[det_idx]
            tr = self.tracks[tid]
            tr.bbox = det.bbox
            tr.cls_name = det.cls_name
            tr.conf = det.conf
            tr.missed = 0

        # Tăng missed nếu không match
        for tid in track_ids:
            if tid not in matches:
                self.tracks[tid].missed += 1

        # Xóa track mất lâu
        to_delete = [tid for tid, tr in self.tracks.items()
                     if tr.missed > self.max_missed]
        for tid in to_delete:
            del self.tracks[tid]

        # Tạo track mới cho detection chưa gán
        for di in unmatched_detections:
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


# ==========================
#  HÀM LÀN ĐƯỜNG
# ==========================

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


# ==========================
#  MAIN
# ==========================

def main():
    print("[INFO] Đang load model YOLOv8...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {VIDEO_SOURCE}")
        return

    target_classes = {"car", "motorbike", "bus", "truck"}
    tracker = SimpleTracker(iou_threshold=0.3, max_missed=10)

    cv2.namedWindow("YOLOv8 Lane Auto Demo", cv2.WINDOW_NORMAL)

    print("[INFO] Bắt đầu xử lý. Nhấn phím 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Hết video hoặc không đọc được frame.")
            break

        h, w = frame.shape[:2]

        # Tạo polygon pixel từ poly_norm
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

        # Gán lane + check vi phạm
        for tr in tracks:
            tr.lane_id = assign_lane_to_track(tr, lanes_pixel)
            tr.lane_type = lane_type_by_id.get(tr.lane_id) if tr.lane_id else None
            tr.is_violation = check_lane_violation(tr)

        # Vẽ làn
        for lane in lanes_pixel:
            pts = np.array(lane["poly"], dtype=np.int32).reshape((-1, 1, 2))
            # màu lane: car = xanh dương, motorbike = vàng nhạt, unknown = trắng
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
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # Vẽ xe + trạng thái
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.bbox)
            color = (0, 255, 0)
            status = "OK"
            if tr.is_violation:
                color = (0, 0, 255)
                status = "VIOLATION"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tr.track_id} {tr.cls_name} L{tr.lane_id} {tr.lane_type} {status}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Thu nhỏ khi hiển thị
        new_w, new_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
        frame_display = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("YOLOv8 Lane Auto Demo", frame_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát.")


if __name__ == "__main__":
    main()
