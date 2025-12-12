from ultralytics import YOLO
import cv2
import os
import numpy as np
import json
from collections import Counter, defaultdict

# ==========================
#  CẤU HÌNH
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data","video", "test 2.mp4")  # đổi nếu cần
OUTPUT_JSON = os.path.join(BASE_DIR, "auto_lanes_config.json")

# Số frame mẫu + bước nhảy
NUM_FRAMES_TO_SAMPLE = 80    # số frame lấy mẫu (cho nhanh, bạn tăng/giảm tùy)
FRAME_STRIDE = 5             # mỗi 5 frame lấy 1 lần

# Chỉ xét nửa dưới khung hình
ROI_Y_MIN_RATIO = 0.5        # 0 = đỉnh ảnh, 1 = đáy ảnh

# Bỏ các bounding box quá nhỏ
MIN_AREA_RATIO = 0.0005      # tỉ lệ so với diện tích ảnh

# Ước lượng số làn (ví dụ video này nhìn là ~3 làn)
ESTIMATED_LANES = 3

# Các class quan tâm
TARGET_CLASSES = {"car", "motorbike", "bus", "truck"}

# Vùng ngang được coi là mặt đường (bỏ lề trái/phải)
ROAD_X_MIN_RATIO = 0.05      # 5% từ mép trái trở vào
ROAD_X_MAX_RATIO = 0.90      # 90% từ bên trái; 10% cuối bên phải coi là vỉa hè/lề


def main():
    print("[INFO] Video source:", VIDEO_SOURCE)
    print("[INFO] Output JSON :", OUTPUT_JSON)

    # ---- Load YOLO ----
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        return

    h, w = frame.shape[:2]
    roi_y_min = int(ROI_Y_MIN_RATIO * h)
    min_area = MIN_AREA_RATIO * w * h

    road_x_min = ROAD_X_MIN_RATIO * w
    road_x_max = ROAD_X_MAX_RATIO * w

    samples_x: list[float] = []
    samples_cls: list[str] = []

    frame_idx = 0
    sampled_frames = 0

    # quay lại đầu video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("[INFO] Collecting samples for lane clustering...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # bỏ qua frame theo stride
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        results = model(frame, conf=0.4)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # chỉ lấy xe trong vùng mặt đường theo X
            if cx < road_x_min or cx > road_x_max:
                continue

            # chỉ lấy nửa dưới khung hình (gần camera)
            if cy < roi_y_min:
                continue

            samples_x.append(float(cx))
            samples_cls.append(cls_name)

        frame_idx += 1
        sampled_frames += 1

        print(f"[INFO] Sampled {sampled_frames}/{NUM_FRAMES_TO_SAMPLE} frames",
              end="\r")

        if sampled_frames >= NUM_FRAMES_TO_SAMPLE:
            break

    cap.release()
    print()  # xuống dòng sau progress

    if not samples_x:
        print("[ERROR] No detections collected for calibration.")
        return

    # ==========================
    #  K-MEANS 1D THEO CX -> LANE
    # ==========================
    X = np.array(samples_x, dtype=np.float32).reshape(-1, 1)
    k = min(ESTIMATED_LANES, len(samples_x))
    print(f"[INFO] Clustering {len(samples_x)} samples into {k} lanes...")

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()  # numpy array

    # chuyển center sang float thuần và sort trái -> phải
    lane_centers_x = sorted([float(c) for c in centers])

    # map cluster index -> lane_id (1..k)
    sorted_idx = np.argsort(centers)
    label_to_lane_id: dict[int, int] = {}
    for lane_id, cluster_index in enumerate(sorted_idx, start=1):
        label_to_lane_id[int(cluster_index)] = lane_id

    # đếm loại xe trong mỗi lane
    lane_class_counts: dict[int, Counter] = defaultdict(Counter)
    for cx, cls_name, cluster_label in zip(samples_x, samples_cls, labels):
        lane_id = label_to_lane_id[int(cluster_label)]
        lane_class_counts[lane_id][cls_name] += 1

    print("[INFO] Lane class distribution:")
    for lane_id in sorted(lane_class_counts.keys()):
        print(f"  Lane {lane_id}:", dict(lane_class_counts[lane_id]))

    # biên giữa các lane (theo X), giới hạn trong vùng mặt đường
    boundaries: list[float] = []
    for i in range(k - 1):
        boundary = (lane_centers_x[i] + lane_centers_x[i + 1]) / 2.0
        boundary = max(road_x_min, min(road_x_max, boundary))
        boundaries.append(float(boundary))

    # ==========================
    #  XÂY JSON CONFIG
    # ==========================
    ny_top = float(ROI_Y_MIN_RATIO)
    ny_bottom = 1.0

    lanes = []
    for idx in range(k):
        lane_id = idx + 1

        if idx == 0:
            x_min = road_x_min
            x_max = boundaries[0] if boundaries else road_x_max
        elif idx == k - 1:
            x_min = boundaries[-1] if boundaries else road_x_min
            x_max = road_x_max
        else:
            x_min = boundaries[idx - 1]
            x_max = boundaries[idx]

        # ép về float chuẩn
        x_min = float(x_min)
        x_max = float(x_max)

        stats = lane_class_counts[lane_id]
        total = int(sum(stats.values()))
        car_like = int(
            stats.get("car", 0)
            + stats.get("bus", 0)
            + stats.get("truck", 0)
        )
        motorbike = int(stats.get("motorbike", 0))

        if total == 0:
            lane_type = "unknown"
        else:
            # nếu >=60% là xe máy -> lane xe máy, ngược lại lane ô tô
            if motorbike / total >= 0.6:
                lane_type = "motorbike"
            else:
                lane_type = "car"

        x_min_norm = x_min / float(w)
        x_max_norm = x_max / float(w)

        lane_cfg = {
            "id": int(lane_id),
            "type": lane_type,  # "car" hoặc "motorbike"
            "poly_norm": [
                [float(x_min_norm), float(ny_top)],
                [float(x_max_norm), float(ny_top)],
                [float(x_max_norm), float(ny_bottom)],
                [float(x_min_norm), float(ny_bottom)],
            ],
        }
        lanes.append(lane_cfg)

    config = {
        "video_source": os.path.basename(VIDEO_SOURCE),
        "frame_width": int(w),
        "frame_height": int(h),
        "roi_y_top_ratio": float(ROI_Y_MIN_RATIO),
        "road_x_min_ratio": float(ROAD_X_MIN_RATIO),
        "road_x_max_ratio": float(ROAD_X_MAX_RATIO),
        "lanes": lanes,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("[INFO] Saved lane config to", OUTPUT_JSON)
    print("[INFO] Lanes:")
    for lane in lanes:
        print(" ", lane)


if __name__ == "__main__":
    main()
