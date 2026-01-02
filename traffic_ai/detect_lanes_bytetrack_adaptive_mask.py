from ultralytics import YOLO
import cv2
import os
import time
import csv
import numpy as np
from typing import List, Tuple, Dict, Optional

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 3.mp4")  # hoặc 0 nếu webcam
YOLO_WEIGHTS = "yolov8n.pt"
TRACKER_CFG = "bytetrack.yaml"

WORK_SCALE = 0.85
DISPLAY_SCALE = 0.85

IMGSZ = 512          # 640 -> 512 (mượt hơn), nếu vẫn lag: 480/416
MAX_DET = 160        # xe máy đông thì để 140~200
CONF_TRACK = 0.08    # để thấp để ít lọt; sẽ lọc lại theo từng lớp ở dưới

# COCO: car=2, motorcycle=3, bus=5, truck=7
TARGET_CLASS_IDS = [2, 3, 5, 7]

# lọc lại theo lớp (giảm lọt, vẫn bắt xe máy)
CAR_MIN_CONF = 0.22
MOTO_MIN_CONF = 0.12
BUS_MIN_CONF = 0.25
TRUCK_MIN_CONF = 0.25

SMOOTH_ALPHA = 0.65

USE_WARP = True
WARP_UPDATE_EVERY = 60     # tăng để hết khựng: 60~90
WARP_SCALE = 0.50          # warp chạy trên ảnh nhỏ
ROI_YMAX = 0.60            # chỉ lấy feature phía trên
MIN_GOOD_MATCHES = 45
MIN_INLIER_RATIO = 0.25
RANSAC_REPROJ = 4.0
SMOOTH_GAMMA = 0.35

DEAD_TRANS_PX = 2.0
DEAD_SCALE = 0.002

# Mask plate: CHỈ xe 4 bánh (car/bus/truck), KHÔNG mask xe máy
MASK_PLATE = True
MASK_MODE = "fill"         # fill nhanh nhất
MASK_NEAR_ONLY = True
MASK_NEAR_Y2_NORM = 0.72
MASK_NEAR_MIN_H = 120

# Vẽ chữ tối ưu
DRAW_LABELS = True
DRAW_LABEL_ONLY_VIOL = True
DRAW_MOTO_LABELS = False   # xe máy chỉ box cho mượt

# Log
SAVE_VIOLATIONS = True
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_CSV = os.path.join(OUTPUT_DIR, "violations_log.csv")

try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# =========================
# ROAD / LANES
# order: left_bottom, right_bottom, right_top, left_top
# =========================
ROAD_POLY_NORM = [
    (0.1224, 0.9111),
    (0.9891, 0.9046),
    (0.7052, 0.4157),
    (0.3443, 0.4343),
]

BOUNDARY_TS = [
    0.0000,
    0.2194,
    0.4742,
    0.7374,
    1.0000,
]

# lane type để quyết định lane nào cho xe máy (nếu muốn)
# ví dụ lane 1 là xe máy: ["motorcycle","car","car","car"]
LANE_TYPES = ["car", "car", "car", "car"]

# =========================
# Helpers
# =========================
def lerp(p1, p2, t: float):
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

def denorm_polygon(poly_norm, w: int, h: int):
    return [(int(x * w), int(y * h)) for x, y in poly_norm]

def generate_lane_polys_pixel(road_poly_px: List[Tuple[int, int]],
                              boundary_ts: List[float],
                              lane_types: List[str]):
    lb, rb, rt, lt = road_poly_px
    ts = sorted([max(0.0, min(1.0, float(t))) for t in boundary_ts])
    n = len(ts) - 1
    if n < 1:
        raise ValueError("BOUNDARY_TS phải có ít nhất 2 giá trị.")
    if len(lane_types) < n:
        lane_types = lane_types + ["car"] * (n - len(lane_types))
    lane_types = lane_types[:n]

    lanes = []
    for i in range(n):
        tL, tR = ts[i], ts[i + 1]
        bL = lerp(lb, rb, tL)
        bR = lerp(lb, rb, tR)
        tR_pt = lerp(lt, rt, tR)
        tL_pt = lerp(lt, rt, tL)
        lanes.append({
            "id": i + 1,
            "name": f"L{i+1}",
            "type": lane_types[i].replace("motorbike", "motorcycle"),
            "poly": [(int(bL[0]), int(bL[1])),
                     (int(bR[0]), int(bR[1])),
                     (int(tR_pt[0]), int(tR_pt[1])),
                     (int(tL_pt[0]), int(tL_pt[1]))]
        })
    return lanes

def build_allowed_lanes(lanes_cfg):
    lanes_by_type: Dict[str, List[int]] = {}
    for lane in lanes_cfg:
        lanes_by_type.setdefault(lane["type"], []).append(lane["id"])

    car_lanes = set(lanes_by_type.get("car", []))
    moto_lanes = set(lanes_by_type.get("motorcycle", []))

    return {
        "car": car_lanes,
        "bus": car_lanes,
        "truck": car_lanes,
        "motorcycle": moto_lanes,
    }

def warp_points(points_px: List[Tuple[int, int]], H: np.ndarray) -> List[Tuple[int, int]]:
    pts = np.float32(points_px).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in warped]

def point_in_poly(px: int, py: int, poly: List[Tuple[int, int]]) -> bool:
    contour = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0

def assign_lane_bottom_center(bbox, lanes_px) -> Optional[int]:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    for lane in lanes_px:
        if point_in_poly(cx, cy, lane["poly"]):
            return lane["id"]
    return None

def class_min_conf(cls_name: str) -> float:
    if cls_name == "car": return CAR_MIN_CONF
    if cls_name == "motorcycle": return MOTO_MIN_CONF
    if cls_name == "bus": return BUS_MIN_CONF
    if cls_name == "truck": return TRUCK_MIN_CONF
    return 0.0

# =========================
# Mask plate (4 bánh, near only)
# =========================
def should_mask(bbox_xyxy, img_h: int, cls_name: str) -> bool:
    if not MASK_PLATE:
        return False
    if cls_name == "motorcycle":
        return False

    x1, y1, x2, y2 = bbox_xyxy
    bh = max(0.0, y2 - y1)

    if not MASK_NEAR_ONLY:
        return True
    return (y2 >= img_h * MASK_NEAR_Y2_NORM) or (bh >= MASK_NEAR_MIN_H)

def mask_plate(img, bbox):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, x1)); y1 = int(max(0, y1))
    x2 = int(min(img.shape[1] - 1, x2)); y2 = int(min(img.shape[0] - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bh < 80:
        return

    px1 = int(x1 + 0.28 * bw)
    px2 = int(x1 + 0.72 * bw)
    py1 = int(y1 + 0.72 * bh)
    py2 = int(y2 - 0.08 * bh)

    px1 = max(0, px1); py1 = max(0, py1)
    px2 = min(img.shape[1], px2); py2 = min(img.shape[0], py2)
    if px2 <= px1 or py2 <= py1:
        return

    if MASK_MODE == "blur":
        roi = img[py1:py2, px1:px2]
        img[py1:py2, px1:px2] = cv2.GaussianBlur(roi, (0, 0), sigmaX=14)
    else:
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 0), -1)

# =========================
# Warp: reference -> current (homography)
# =========================
def homography_motion_stats(H: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    Hn = H / (H[2, 2] + 1e-8)
    c = np.array([[w * 0.5, h * 0.5, 1.0]], dtype=np.float32).T
    p = np.array([[w * 0.5 + 120.0, h * 0.5, 1.0]], dtype=np.float32).T

    c2 = Hn @ c
    p2 = Hn @ p
    c2 = (c2[:2] / (c2[2] + 1e-8)).reshape(2)
    p2 = (p2[:2] / (p2[2] + 1e-8)).reshape(2)

    trans = float(np.linalg.norm(c2 - np.array([w * 0.5, h * 0.5], dtype=np.float32)))
    dist0 = 120.0
    dist1 = float(np.linalg.norm(p2 - c2))
    scale = dist1 / max(1e-6, dist0)
    return trans, scale

class ReferenceHomographyWarp:
    def __init__(self, ref_frame_bgr, road_poly_px, roi_ymax=0.60, warp_scale=0.50,
                 min_good=45, min_inlier=0.25, ransac_reproj=4.0, smooth_gamma=0.35):
        self.warp_scale = float(warp_scale)
        self.min_good = int(min_good)
        self.min_inlier = float(min_inlier)
        self.ransac_reproj = float(ransac_reproj)
        self.smooth_gamma = float(smooth_gamma)

        self.det = cv2.AKAZE_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.fail_count = 0
        self.last_good = True
        self.last_inlier = 0.0
        self.last_good_matches = 0

        ref_gray_full = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = ref_gray_full.shape[:2]

        mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_full, (0, 0), (w, int(h * roi_ymax)), 255, -1)
        road_cnt = np.array(road_poly_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask_full, [road_cnt], 0)

        if self.warp_scale != 1.0:
            self.ref_gray = cv2.resize(ref_gray_full, None, fx=self.warp_scale, fy=self.warp_scale,
                                       interpolation=cv2.INTER_AREA)
            self.mask = cv2.resize(mask_full, (self.ref_gray.shape[1], self.ref_gray.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            self.ref_gray = ref_gray_full
            self.mask = mask_full

        self.ref_kp, self.ref_des = self.det.detectAndCompute(self.ref_gray, self.mask)
        self.H = np.eye(3, dtype=np.float32)

    def update(self, frame_bgr, full_w: int, full_h: int):
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.warp_scale != 1.0:
            gray = cv2.resize(gray_full, None, fx=self.warp_scale, fy=self.warp_scale,
                              interpolation=cv2.INTER_AREA)
        else:
            gray = gray_full

        kp, des = self.det.detectAndCompute(gray, self.mask)
        if des is None or self.ref_des is None or len(kp) < 60 or len(self.ref_kp) < 60:
            self.fail_count += 1
            self.last_good = False
            return self.H, False

        matches = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        self.last_good_matches = len(good)
        if len(good) < self.min_good:
            self.fail_count += 1
            self.last_good = False
            return self.H, False

        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H_new, inliers = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_reproj)
        if H_new is None or inliers is None:
            self.fail_count += 1
            self.last_good = False
            return self.H, False

        inlier_ratio = float(inliers.mean())
        self.last_inlier = inlier_ratio
        if inlier_ratio < self.min_inlier:
            self.fail_count += 1
            self.last_good = False
            return self.H, False

        H_new = H_new.astype(np.float32)

        if self.warp_scale != 1.0:
            s = self.warp_scale
            S = np.array([[s, 0, 0],
                          [0, s, 0],
                          [0, 0, 1]], dtype=np.float32)
            Sinv = np.array([[1/s, 0, 0],
                             [0, 1/s, 0],
                             [0, 0, 1]], dtype=np.float32)
            H_new = Sinv @ H_new @ S

        H_new = H_new / (H_new[2, 2] + 1e-8)

        trans, sc = homography_motion_stats(H_new, full_w, full_h)
        if trans < DEAD_TRANS_PX and abs(sc - 1.0) < DEAD_SCALE:
            self.fail_count = 0
            self.last_good = True
            return self.H, True

        H_old = self.H / (self.H[2, 2] + 1e-8)
        H_blend = (1.0 - self.smooth_gamma) * H_old + self.smooth_gamma * H_new
        H_blend = H_blend / (H_blend[2, 2] + 1e-8)

        self.H = H_blend
        self.fail_count = 0
        self.last_good = True
        return self.H, True

# =========================
# EMA bbox smoother
# =========================
class BoxSmoother:
    def __init__(self, alpha=0.65):
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.state: Dict[int, np.ndarray] = {}

    def update(self, track_id: int, bbox_xyxy: np.ndarray) -> np.ndarray:
        bbox = bbox_xyxy.astype(np.float32)
        if track_id not in self.state:
            self.state[track_id] = bbox
            return bbox
        prev = self.state[track_id]
        sm = self.alpha * bbox + (1.0 - self.alpha) * prev
        self.state[track_id] = sm
        return sm

# =========================
# MAIN
# =========================
def main():
    log_f = None
    log_w = None
    if SAVE_VIOLATIONS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        new_file = not os.path.exists(LOG_CSV)
        log_f = open(LOG_CSV, "a", newline="", encoding="utf-8")
        log_w = csv.writer(log_f)
        if new_file:
            log_w.writerow(["ts", "frame", "track_id", "class", "conf", "lane_id", "status"])

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Cannot open:", VIDEO_SOURCE)
        return

    ret, f0 = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        return

    if WORK_SCALE != 1.0:
        f0 = cv2.resize(f0, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)

    Hh, Ww = f0.shape[:2]

    road_ref_px = denorm_polygon(ROAD_POLY_NORM, Ww, Hh)
    lanes_ref_px = generate_lane_polys_pixel(road_ref_px, BOUNDARY_TS, LANE_TYPES)
    allowed_lanes_by_cls = build_allowed_lanes(lanes_ref_px)

    warp = ReferenceHomographyWarp(
        f0, road_ref_px,
        roi_ymax=ROI_YMAX,
        warp_scale=WARP_SCALE,
        min_good=MIN_GOOD_MATCHES,
        min_inlier=MIN_INLIER_RATIO,
        ransac_reproj=RANSAC_REPROJ,
        smooth_gamma=SMOOTH_GAMMA
    ) if USE_WARP else None

    smoother = BoxSmoother(alpha=SMOOTH_ALPHA)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(YOLO_WEIGHTS)
    try:
        model.fuse()
    except Exception:
        pass

    cv2.namedWindow("ITS Stream", cv2.WINDOW_NORMAL)

    frame_idx = 0
    fps_ema = 0.0
    last_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if WORK_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)

        frame_idx += 1
        out = frame.copy()

        H = np.eye(3, dtype=np.float32)
        warp_ok = True
        if USE_WARP and warp is not None:
            if (frame_idx % WARP_UPDATE_EVERY) == 0:
                H, warp_ok = warp.update(frame, Ww, Hh)
            else:
                H = warp.H
                warp_ok = warp.last_good
        else:
            warp_ok = True

        road_cur_px = warp_points(road_ref_px, H)
        lanes_cur_px = [{
            "id": ln["id"],
            "name": ln["name"],
            "type": ln["type"],
            "poly": warp_points(ln["poly"], H),
        } for ln in lanes_ref_px]

        cv2.polylines(out, [np.array(road_cur_px, np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2)
        for lane in lanes_cur_px:
            color_lane = (255, 0, 0) if lane["type"] == "car" else (0, 255, 255)
            cv2.polylines(out, [np.array(lane["poly"], np.int32).reshape(-1, 1, 2)], True, color_lane, 2)

        t_y0 = time.time()
        res = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_TRACK,
            classes=TARGET_CLASS_IDS,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            verbose=False
        )[0]
        yolo_ms = int((time.time() - t_y0) * 1000)

        boxes = res.boxes
        if boxes is None or boxes.xyxy is None:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=int)
            confs = np.zeros((0,), dtype=np.float32)
            ids = None
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
            ids = None if boxes.id is None else boxes.id.cpu().numpy().astype(int)

        violations = 0
        kept = 0

        for i in range(len(xyxy)):
            cls_id = int(cls[i])
            cls_name = model.names.get(cls_id, str(cls_id))
            if cls_name == "motorbike":
                cls_name = "motorcycle"
            c = float(confs[i])

            if c < class_min_conf(cls_name):
                continue

            kept += 1
            tid = int(ids[i]) if ids is not None else (i + 1)

            bbox_sm = smoother.update(tid, xyxy[i])
            x1, y1, x2, y2 = [float(v) for v in bbox_sm]

            lane_id = assign_lane_bottom_center((x1, y1, x2, y2), lanes_cur_px)
            allowed = allowed_lanes_by_cls.get(cls_name, set())
            is_viol = (lane_id is not None) and (len(allowed) > 0) and (lane_id not in allowed)
            if is_viol:
                violations += 1

            if should_mask((x1, y1, x2, y2), img_h=Hh, cls_name=cls_name):
                mask_plate(out, (x1, y1, x2, y2))

            color = (0, 0, 255) if is_viol else (0, 255, 0)
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            ix1 = max(0, min(Ww - 1, ix1))
            iy1 = max(0, min(Hh - 1, iy1))
            ix2 = max(0, min(Ww - 1, ix2))
            iy2 = max(0, min(Hh - 1, iy2))
            if ix2 > ix1 and iy2 > iy1:
                cv2.rectangle(out, (ix1, iy1), (ix2, iy2), color, 2)

                if DRAW_LABELS:
                    if (not DRAW_LABEL_ONLY_VIOL) or is_viol:
                        if (cls_name != "motorcycle") or DRAW_MOTO_LABELS:
                            lane_txt = f" L{lane_id}" if lane_id is not None else ""
                            cv2.putText(out, f"{cls_name}:{tid}{lane_txt}",
                                        (ix1, max(0, iy1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            if SAVE_VIOLATIONS and is_viol and log_w is not None:
                log_w.writerow([time.time(), frame_idx, tid, cls_name, f"{c:.3f}", lane_id, "VIOLATION"])

        now = time.time()
        dt = max(1e-6, now - last_t)
        inst_fps = 1.0 / dt
        fps_ema = inst_fps if fps_ema <= 0 else (0.15 * inst_fps + 0.85 * fps_ema)
        last_t = now

        inl = warp.last_inlier if (USE_WARP and warp is not None) else 0.0
        goodm = warp.last_good_matches if (USE_WARP and warp is not None) else 0

        cv2.putText(out,
                    f"fps={fps_ema:.1f} yolo={yolo_ms}ms kept={kept} viol={violations} scale={WORK_SCALE} imgsz={IMGSZ}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out,
                    f"warp={'OK' if warp_ok else 'HOLD'} inl={inl:.2f} good={goodm} every={WARP_UPDATE_EVERY}",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        disp = out
        if DISPLAY_SCALE != 1.0:
            disp = cv2.resize(out, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("ITS Stream", disp)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if log_f is not None:
        log_f.close()

if __name__ == "__main__":
    main()
