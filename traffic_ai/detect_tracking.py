from ultralytics import YOLO
import cv2
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa file .py
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 1.mp4")
DISPLAY_SCALE = 0.6  # thu nhỏ khi hiển thị

print("[DEBUG] VIDEO_SOURCE =", VIDEO_SOURCE)
print("[DEBUG] File tồn tại? :", os.path.exists(VIDEO_SOURCE))



@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    cls_name: str
    conf: float


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    cls_name: str
    conf: float
    missed: int = 0  # số frame không match (để xóa track cũ)


def iou(box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]) -> float:
    """Tính IoU giữa 2 bbox."""
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
    """Tracker đơn giản dựa trên IoU, đủ dùng cho demo ITS."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        # Nếu chưa có track nào, tạo mới hết
        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())

        # Gán detection cho track dựa trên IoU cao nhất
        unmatched_detections = list(range(len(detections)))
        track_ids = list(self.tracks.keys())

        # Lưu match (track_id -> det_index)
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

        # Cập nhật track đã match
        for tid, det_idx in matches.items():
            det = detections[det_idx]
            tr = self.tracks[tid]
            tr.bbox = det.bbox
            tr.cls_name = det.cls_name
            tr.conf = det.conf
            tr.missed = 0

        # Tăng missed cho track không match
        for tid in track_ids:
            if tid not in matches:
                self.tracks[tid].missed += 1

        # Xóa track quá lâu không thấy
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

 

def main():
    print("[INFO] Đang load model YOLOv8...")
    model = YOLO("yolov8n.pt")  # hoặc đường dẫn tới model custom

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được nguồn video: {VIDEO_SOURCE}")
        return

    target_classes = {"car", "motorbike", "bus", "truck"}
    tracker = SimpleTracker(iou_threshold=0.3, max_missed=10)

    cv2.namedWindow("YOLOv8 Tracking Demo", cv2.WINDOW_NORMAL)

    print("[INFO] Bắt đầu xử lý. Nhấn phím 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Hết video hoặc không đọc được frame.")
            break

        # ---- YOLO DETECTION ----
        results = model(frame, conf=0.4)[0]

        detections: List[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name not in target_classes:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    cls_name=cls_name,
                    conf=conf,
                )
            )

        # ---- TRACKING ----
        tracks = tracker.update(detections)

        # ---- VẼ KẾT QUẢ ----
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {tr.track_id} {tr.cls_name} {tr.conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ---- THU NHỎ KHI HIỂN THỊ ----
        h, w = frame.shape[:2]
        new_w, new_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
        frame_display = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow("YOLOv8 Tracking Demo", frame_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát.")


if __name__ == "__main__":
    main()
