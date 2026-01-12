import os
import time, csv
import cv2
import numpy as np
from collections import deque
import threading
import queue

CPU_CORES = os.cpu_count() or 4
N_THREADS = max(1, CPU_CORES - 1)

os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)

try:
    cv2.setUseOptimized(True)
except Exception:
    pass
try:
    cv2.setNumThreads(N_THREADS)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(N_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
except Exception:
    torch = None

from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 3.mp4")

YOLO_WEIGHTS = "yolov8s.pt"
TRACKER_CFG = "bytetrack.yaml"

WORK_SCALE = 1.00
DISPLAY_SCALE = 0.80
IMGSZ = 640
MAX_DET = 200
CONF_TRACK = 0.08
TARGET_CLASS_IDS = [2, 3, 5, 7]
VID_STRIDE = 1

ENABLE_FAR_PASS = True
FAR_EVERY = 6
FAR_TOP_FRAC = 0.65
FAR_IMGSZ = 768
FAR_CONF = 0.05
FAR_MAX_DET = 250
FAR_IOU_SUPPRESS = 0.50
FAR_AUGMENT = False

CAR_MIN_CONF = 0.10
MOTO_MIN_CONF = 0.05
BUS_MIN_CONF = 0.12
TRUCK_MIN_CONF = 0.12

ROAD_POLY_NORM = [
    (0.1208, 0.9176),
    (0.9938, 0.9157),
    (0.6885, 0.3519),
    (0.3672, 0.3556),
]
BOUNDARY_TS = [0.0000, 0.2194, 0.4742, 0.7374, 1.0000]
LANE_TYPES = ["motorcycle", "motorcycle", "car", "car"]

USE_WARP = True
WARP_UPDATE_EVERY = 90
WARP_SCALE = 0.50
ROI_YMAX = 0.60
MIN_GOOD_MATCHES = 45
MIN_INLIER_RATIO = 0.25
RANSAC_REPROJ = 4.0
SMOOTH_GAMMA = 0.35
DEAD_TRANS_PX = 2.0
DEAD_SCALE = 0.002

DETECT_ROI = True
ROI_PAD_X = 0.03
ROI_PAD_Y = 0.35

MASK_PLATE = True
MASK_NEAR_Y2_NORM = 0.72
MASK_NEAR_MIN_H = 120

SAVE_VIOLATIONS = True
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_CSV = os.path.join(OUTPUT_DIR, "violations_log.csv")

DRAW_TEXT = False

ENABLE_PROFILER = True
PROF_PRINT_EVERY_SEC = 0.25
PROF_SMOOTH_N = 30


def lerp(p1, p2, t):
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

def denorm_polygon(poly_norm, w, h):
    return [(int(x * w), int(y * h)) for x, y in poly_norm]

def generate_lane_polys_pixel(road_poly_px, boundary_ts, lane_types):
    lb, rb, rt, lt = road_poly_px
    ts = sorted([max(0.0, min(1.0, float(t))) for t in boundary_ts])
    n = len(ts) - 1
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
            "type": lane_types[i].replace("motorbike", "motorcycle"),
            "poly": [(int(bL[0]), int(bL[1])),
                     (int(bR[0]), int(bR[1])),
                     (int(tR_pt[0]), int(tR_pt[1])),
                     (int(tL_pt[0]), int(tL_pt[1]))]
        })
    return lanes

def build_allowed_lanes(lanes_cfg):
    lanes_by_type = {}
    for lane in lanes_cfg:
        lanes_by_type.setdefault(lane["type"], []).append(lane["id"])
    car_lanes = set(lanes_by_type.get("car", []))
    moto_lanes = set(lanes_by_type.get("motorcycle", []))
    return {"car": car_lanes, "bus": car_lanes, "truck": car_lanes, "motorcycle": moto_lanes}

def warp_points(points_px, H):
    pts = np.float32(points_px).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in warped]

def point_in_poly(px, py, poly):
    contour = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0

def assign_lane_bottom_center(bbox, lanes_px):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    for lane in lanes_px:
        if point_in_poly(cx, cy, lane["poly"]):
            return lane["id"]
    return None

def class_min_conf(cls_name):
    if cls_name == "car": return CAR_MIN_CONF
    if cls_name == "motorcycle": return MOTO_MIN_CONF
    if cls_name == "bus": return BUS_MIN_CONF
    if cls_name == "truck": return TRUCK_MIN_CONF
    return 0.0

def should_mask(bbox_xyxy, img_h, cls_name):
    if not MASK_PLATE:
        return False
    if cls_name == "motorcycle":
        return False
    x1, y1, x2, y2 = bbox_xyxy
    bh = max(0.0, y2 - y1)
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
    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 0), -1)

def roi_from_poly(poly_px, w, h, pad_x=0.03, pad_y=0.35):
    xs = [p[0] for p in poly_px]
    ys = [p[1] for p in poly_px]
    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
    dx = int((x2 - x1) * pad_x)
    dy = int((y2 - y1) * pad_y)
    x1 = max(0, x1 - dx); x2 = min(w - 1, x2 + dx)
    y1 = max(0, y1 - dy); y2 = min(h - 1, y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2

def homography_motion_gate(H, w, h):
    Hn = H / (H[2, 2] + 1e-8)
    c = np.array([[w * 0.5, h * 0.5, 1.0]], dtype=np.float32).T
    p = np.array([[w * 0.5 + 120.0, h * 0.5, 1.0]], dtype=np.float32).T
    c2 = Hn @ c
    p2 = Hn @ p
    c2 = (c2[:2] / (c2[2] + 1e-8)).reshape(2)
    p2 = (p2[:2] / (p2[2] + 1e-8)).reshape(2)
    trans = float(np.linalg.norm(c2 - np.array([w * 0.5, h * 0.5], dtype=np.float32)))
    scale = float(np.linalg.norm(p2 - c2)) / 120.0
    return (trans < DEAD_TRANS_PX) and (abs(scale - 1.0) < DEAD_SCALE)

class ReferenceHomographyWarp:
    def __init__(self, ref_frame_bgr, road_poly_px):
        self.det = cv2.AKAZE_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last_good = True
        self.last_inlier = 0.0
        self.last_good_matches = 0
        self.H = np.eye(3, dtype=np.float32)

        ref_gray_full = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = ref_gray_full.shape[:2]

        mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_full, (0, 0), (w, int(h * ROI_YMAX)), 255, -1)
        road_cnt = np.array(road_poly_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask_full, [road_cnt], 0)

        self.ref_gray = cv2.resize(ref_gray_full, None, fx=WARP_SCALE, fy=WARP_SCALE, interpolation=cv2.INTER_AREA)
        self.mask = cv2.resize(mask_full, (self.ref_gray.shape[1], self.ref_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.ref_kp, self.ref_des = self.det.detectAndCompute(self.ref_gray, self.mask)

    def update(self, frame_bgr, full_w, full_h):
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_full, None, fx=WARP_SCALE, fy=WARP_SCALE, interpolation=cv2.INTER_AREA)

        kp, des = self.det.detectAndCompute(gray, self.mask)
        if des is None or self.ref_des is None or len(kp) < 60 or len(self.ref_kp) < 60:
            self.last_good = False
            return self.H, False

        matches = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        self.last_good_matches = len(good)
        if len(good) < MIN_GOOD_MATCHES:
            self.last_good = False
            return self.H, False

        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H_new, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
        if H_new is None or inliers is None:
            self.last_good = False
            return self.H, False

        inlier_ratio = float(inliers.mean())
        self.last_inlier = inlier_ratio
        if inlier_ratio < MIN_INLIER_RATIO:
            self.last_good = False
            return self.H, False

        H_new = H_new.astype(np.float32)

        s = WARP_SCALE
        S = np.array([[s, 0, 0],[0, s, 0],[0, 0, 1]], dtype=np.float32)
        Sinv = np.array([[1/s, 0, 0],[0, 1/s, 0],[0, 0, 1]], dtype=np.float32)
        H_new = Sinv @ H_new @ S
        H_new = H_new / (H_new[2, 2] + 1e-8)

        if homography_motion_gate(H_new, full_w, full_h):
            self.last_good = True
            return self.H, True

        H_old = self.H / (H_old := (self.H[2, 2] + 1e-8))
        H_blend = (1.0 - SMOOTH_GAMMA) * H_old + SMOOTH_GAMMA * H_new
        H_blend = H_blend / (H_blend[2, 2] + 1e-8)

        self.H = H_blend
        self.last_good = True
        return self.H, True

def pick_device():
    if torch is None:
        return "cpu", False
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0", True
    except Exception:
        pass
    return "cpu", False

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1 = b[:, 0]; by1 = b[:, 1]; bx2 = b[:, 2]; by2 = b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, inter_x2 - inter_x1)
    ih = np.maximum(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return inter / union

def on_road_bottom_center(bbox_xyxy, road_cnt):
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) * 0.5
    cy = y2
    return cv2.pointPolygonTest(road_cnt, (float(cx), float(cy)), False) >= 0


class FrameQueue:
    def __init__(self, maxsize=4):
        self.q = queue.Queue(maxsize=maxsize)
        self.ended = False

    def push(self, frame, idx):
        if self.ended:
            return
        self.q.put((idx, frame))

    def pop(self, stop_event):
        while not stop_event.is_set():
            try:
                return self.q.get(timeout=0.05)
            except queue.Empty:
                if self.ended:
                    return None, None
        return None, None

    def set_ended(self):
        self.ended = True


def capture_loop(cap, fq: FrameQueue, stop_event: threading.Event):
    idx = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            fq.set_ended()
            break

        if VID_STRIDE > 1:
            for _ in range(VID_STRIDE - 1):
                if not cap.grab():
                    break

        if WORK_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)

        idx += 1
        fq.push(frame, idx)


def main():
    log_f, log_w = None, None
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
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

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

    warp = ReferenceHomographyWarp(f0, road_ref_px) if USE_WARP else None

    model = YOLO(YOLO_WEIGHTS)
    model_far = YOLO(YOLO_WEIGHTS)
    try:
        model.fuse()
        model_far.fuse()
    except Exception:
        pass

    device, use_half = pick_device()
    print(f"[INFO] device={device} half={use_half} threads={N_THREADS}")

    cv2.namedWindow("ITS Stream", cv2.WINDOW_NORMAL)

    fq = FrameQueue(maxsize=4)
    stop_event = threading.Event()

    tcap = threading.Thread(target=capture_loop, args=(cap, fq, stop_event), daemon=True)
    tcap.start()

    fps_ema = 0.0
    last_t = time.perf_counter()

    total_hist = deque(maxlen=PROF_SMOOTH_N)
    last_prof_print = time.perf_counter()

    far_cache_xyxy = np.zeros((0, 4), dtype=np.float32)
    far_cache_cls = np.zeros((0,), dtype=int)
    far_cache_conf = np.zeros((0,), dtype=np.float32)
    far_cache_age = 0

    frame_idx = 0

    while True:
        t0 = time.perf_counter()

        frame_idx, frame = fq.pop(stop_event)
        if frame is None:
            break

        out = frame.copy()

        t_warp0 = time.perf_counter()
        H = np.eye(3, dtype=np.float32)
        warp_ok = True
        if USE_WARP and warp is not None:
            if (frame_idx % WARP_UPDATE_EVERY) == 0:
                H, warp_ok = warp.update(frame, Ww, Hh)
            else:
                H = warp.H
                warp_ok = warp.last_good
        t_warp_ms = (time.perf_counter() - t_warp0) * 1000.0

        road_cur_px = warp_points(road_ref_px, H)
        road_cnt = np.array(road_cur_px, dtype=np.int32).reshape(-1, 1, 2)
        lanes_cur_px = [{"id": ln["id"], "type": ln["type"], "poly": warp_points(ln["poly"], H)} for ln in lanes_ref_px]

        cv2.polylines(out, [road_cnt], True, (255, 255, 0), 2)
        for lane in lanes_cur_px:
            col = (255, 0, 0) if lane["type"] == "car" else (0, 255, 255)
            cv2.polylines(out, [np.array(lane["poly"], np.int32).reshape(-1, 1, 2)], True, col, 2)

        t_roi0 = time.perf_counter()
        if DETECT_ROI:
            rx1, ry1, rx2, ry2 = roi_from_poly(road_cur_px, Ww, Hh, ROI_PAD_X, ROI_PAD_Y)
            det_img = frame[ry1:ry2, rx1:rx2]
            offx, offy = rx1, ry1
        else:
            det_img = frame
            offx, offy = 0, 0
            rx1, ry1, rx2, ry2 = 0, 0, Ww, Hh
        t_roi_ms = (time.perf_counter() - t_roi0) * 1000.0

        t_y0 = time.perf_counter()
        track_kwargs = dict(
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_TRACK,
            classes=TARGET_CLASS_IDS,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            verbose=False,
            device=device,
        )
        if use_half and device != "cpu":
            track_kwargs["half"] = True

        res = model.track(det_img, **track_kwargs)[0]
        yolo_ms = (time.perf_counter() - t_y0) * 1000.0

        boxes = res.boxes
        if boxes is None or boxes.xyxy is None:
            near_xyxy = np.zeros((0, 4), dtype=np.float32)
            near_cls = np.zeros((0,), dtype=int)
            near_conf = np.zeros((0,), dtype=np.float32)
            near_ids = None
        else:
            near_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            near_cls = boxes.cls.cpu().numpy().astype(int)
            near_conf = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.ones((len(near_xyxy),), dtype=np.float32)
            near_ids = None if boxes.id is None else boxes.id.cpu().numpy().astype(int)

        if len(near_xyxy) > 0:
            near_xyxy[:, [0, 2]] += float(offx)
            near_xyxy[:, [1, 3]] += float(offy)

        t_far_ms = 0.0
        if ENABLE_FAR_PASS:
            far_cache_age += 1

            roi_h = max(1, (ry2 - ry1))
            far_y1 = ry1
            far_y2 = ry1 + int(roi_h * FAR_TOP_FRAC)
            far_x1 = rx1
            far_x2 = rx2

            far_y1 = max(0, min(Hh - 1, far_y1))
            far_y2 = max(0, min(Hh, far_y2))
            far_x1 = max(0, min(Ww - 1, far_x1))
            far_x2 = max(0, min(Ww, far_x2))

            do_far = (frame_idx % FAR_EVERY) == 0
            if do_far and far_y2 > far_y1 and far_x2 > far_x1:
                t_f0 = time.perf_counter()
                far_img = frame[far_y1:far_y2, far_x1:far_x2]

                pred_kwargs = dict(
                    imgsz=FAR_IMGSZ,
                    conf=FAR_CONF,
                    classes=TARGET_CLASS_IDS,
                    max_det=FAR_MAX_DET,
                    device=device,
                    verbose=False,
                    augment=FAR_AUGMENT,
                )
                if use_half and device != "cpu":
                    pred_kwargs["half"] = True

                far_res = model_far.predict(far_img, **pred_kwargs)[0]
                t_far_ms = (time.perf_counter() - t_f0) * 1000.0

                fboxes = far_res.boxes
                if fboxes is None or fboxes.xyxy is None:
                    far_xyxy = np.zeros((0, 4), dtype=np.float32)
                    far_cls = np.zeros((0,), dtype=int)
                    far_conf = np.zeros((0,), dtype=np.float32)
                else:
                    far_xyxy = fboxes.xyxy.cpu().numpy().astype(np.float32)
                    far_cls = fboxes.cls.cpu().numpy().astype(int)
                    far_conf = fboxes.conf.cpu().numpy().astype(np.float32) if fboxes.conf is not None else np.ones((len(far_xyxy),), dtype=np.float32)

                if len(far_xyxy) > 0:
                    far_xyxy[:, [0, 2]] += float(far_x1)
                    far_xyxy[:, [1, 3]] += float(far_y1)

                    keep = []
                    for k in range(len(far_xyxy)):
                        if on_road_bottom_center(far_xyxy[k], road_cnt):
                            keep.append(k)
                    if keep:
                        far_xyxy = far_xyxy[keep]
                        far_cls = far_cls[keep]
                        far_conf = far_conf[keep]
                    else:
                        far_xyxy = np.zeros((0, 4), dtype=np.float32)
                        far_cls = np.zeros((0,), dtype=int)
                        far_conf = np.zeros((0,), dtype=np.float32)

                far_cache_xyxy = far_xyxy
                far_cache_cls = far_cls
                far_cache_conf = far_conf
                far_cache_age = 0

        if far_cache_age > max(2, FAR_EVERY * 2):
            far_cache_xyxy = np.zeros((0, 4), dtype=np.float32)
            far_cache_cls = np.zeros((0,), dtype=int)
            far_cache_conf = np.zeros((0,), dtype=np.float32)

        merged_xyxy = near_xyxy
        merged_cls = near_cls
        merged_conf = near_conf
        merged_ids = near_ids

        if ENABLE_FAR_PASS and len(far_cache_xyxy) > 0:
            keep_far_idx = []
            if len(near_xyxy) == 0:
                keep_far_idx = list(range(len(far_cache_xyxy)))
            else:
                for j in range(len(far_cache_xyxy)):
                    ious = iou_xyxy(far_cache_xyxy[j], near_xyxy)
                    if float(ious.max()) < FAR_IOU_SUPPRESS:
                        keep_far_idx.append(j)

            if keep_far_idx:
                far_k_xyxy = far_cache_xyxy[keep_far_idx]
                far_k_cls = far_cache_cls[keep_far_idx]
                far_k_conf = far_cache_conf[keep_far_idx]

                merged_xyxy = np.concatenate([merged_xyxy, far_k_xyxy], axis=0) if len(merged_xyxy) else far_k_xyxy
                merged_cls = np.concatenate([merged_cls, far_k_cls], axis=0) if len(merged_cls) else far_k_cls
                merged_conf = np.concatenate([merged_conf, far_k_conf], axis=0) if len(merged_conf) else far_k_conf

                far_ids = -100000 - np.arange(len(far_k_xyxy), dtype=int)
                if merged_ids is None and len(near_xyxy) > 0:
                    merged_ids = np.arange(1, len(near_xyxy) + 1, dtype=int)
                if merged_ids is None:
                    merged_ids = far_ids
                else:
                    merged_ids = np.concatenate([merged_ids, far_ids], axis=0)

        t_post0 = time.perf_counter()
        violations = 0
        kept = 0

        for i in range(len(merged_xyxy)):
            x1, y1, x2, y2 = merged_xyxy[i]
            if not on_road_bottom_center((x1, y1, x2, y2), road_cnt):
                continue

            cls_id = int(merged_cls[i])
            cls_name = model.names.get(cls_id, str(cls_id))
            if cls_name == "motorbike":
                cls_name = "motorcycle"
            c = float(merged_conf[i])

            if c < class_min_conf(cls_name):
                continue

            kept += 1
            tid = int(merged_ids[i]) if merged_ids is not None else (i + 1)

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
                if DRAW_TEXT and is_viol:
                    cv2.putText(out, f"{cls_name}:{tid} L{lane_id}",
                                (ix1, max(0, iy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            if SAVE_VIOLATIONS and is_viol and log_w is not None:
                log_w.writerow([time.time(), frame_idx, tid, cls_name, f"{c:.3f}", lane_id, "VIOLATION"])

        t_post_ms = (time.perf_counter() - t_post0) * 1000.0

        now = time.perf_counter()
        dt = max(1e-6, now - last_t)
        inst_fps = 1.0 / dt
        fps_ema = inst_fps if fps_ema <= 0 else (0.15 * inst_fps + 0.85 * fps_ema)
        last_t = now

        inl = warp.last_inlier if (USE_WARP and warp is not None) else 0.0
        goodm = warp.last_good_matches if (USE_WARP and warp is not None) else 0

        cv2.putText(out,
                    f"fps={fps_ema:.1f} near={yolo_ms:.0f}ms far={t_far_ms:.0f}ms kept={kept} viol={violations}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out,
                    f"warp={'OK' if warp_ok else 'HOLD'} inl={inl:.2f} good={goodm} farEvery={FAR_EVERY}",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        t_disp0 = time.perf_counter()
        disp = out
        if DISPLAY_SCALE != 1.0:
            disp = cv2.resize(out, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("ITS Stream", disp)
        t_disp_ms = (time.perf_counter() - t_disp0) * 1000.0

        t_total_ms = (time.perf_counter() - t0) * 1000.0
        total_hist.append(t_total_ms)
        avg_ms = sum(total_hist) / len(total_hist)
        fps_est = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        if ENABLE_PROFILER and (time.perf_counter() - last_prof_print) >= PROF_PRINT_EVERY_SEC:
            print(
                f"ms total={t_total_ms:.1f}(avg={avg_ms:.1f},FPS~{fps_est:.1f}) "
                f"| warp={t_warp_ms:.1f} roi={t_roi_ms:.1f} near={yolo_ms:.1f} far={t_far_ms:.1f} post={t_post_ms:.1f} disp={t_disp_ms:.1f}",
                end="\r"
            )
            last_prof_print = time.perf_counter()

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            stop_event.set()
            break

    stop_event.set()
    try:
        tcap.join(timeout=1.0)
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
    if log_f is not None:
        log_f.close()
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
