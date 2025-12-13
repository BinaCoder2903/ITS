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
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 1.mp4")  # đổi video
YOLO_WEIGHTS = "yolov8n.pt"
TRACKER_CFG = "bytetrack.yaml"

# Speed
WORK_SCALE = 0.85      # scale frame để tăng fps (0.7~1.0)
DISPLAY_SCALE = 0.85   # scale cửa sổ hiển thị
IMGSZ = 480            # 640 -> 512 -> 480 nếu lag
CONF_TH = 0.35
MAX_DET = 60
TARGET_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)

# Track smoothing (EMA bbox) 0..1 (càng lớn càng bám bbox mới)
SMOOTH_ALPHA = 0.65

# Warp chống lệch lane do zoom/rung camera
USE_WARP = True
WARP_UPDATE_EVERY = 10   # 5~15 ok
WARP_SCALE = 0.5         # chạy ORB trên ảnh nhỏ để nhanh (0.4~0.6)
ROI_YMAX = 0.60

MIN_GOOD_MATCHES = 70
MIN_INLIER_RATIO = 0.25
RANSAC_REPROJ = 4.0
REBASE_FAILS = 35        # fail liên tục thì rebase

# Deadband: camera đứng yên => lane đứng yên (đỡ rung / trôi)
DEADBAND_PX = 3.0        # tịnh tiến < px => coi như đứng yên
DEADBAND_SCALE = 0.004   # |scale-1| nhỏ => coi như đứng yên
DEADBAND_DEG = 0.35      # góc xoay nhỏ => coi như đứng yên
WARP_EMA = 0.25          # smoothing warp (0.2~0.35)

MASK_ROAD_IN_WARP = True  # mask vùng road khi warp để tránh xe chạy

# Mask biển số (public stream)
MASK_PLATE = True
MASK_MODE = "fill"        # "fill" nhanh; "blur" đẹp nhưng nặng
MASK_NEAR_ONLY = True
MASK_NEAR_Y2_NORM = 0.72  # chỉ che nếu đáy bbox ở dưới % chiều cao
MASK_NEAR_MIN_H = 120     # hoặc bbox cao >= px

# Log vi phạm (nếu muốn)
SAVE_VIOLATIONS = True
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_CSV = os.path.join(OUTPUT_DIR, "violations_log.csv")

# giảm giật do thread
try:
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

# =========================
# ROAD/LANE (NORM 0..1)
# order: left_bottom, right_bottom, right_top, left_top
# =========================
ROAD_POLY_NORM = [
    (0.2250, 0.9583),
    (0.7417, 0.9704),
    (0.5620, 0.4481),
    (0.4354, 0.4574),
]



BOUNDARY_TS = [0.0000, 0.3224, 0.6289, 1.0000]
LANE_TYPES = ["car", "car", "car"]  # ví dụ ["car","car","motorcycle"]

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

        lane_type = lane_types[i].replace("motorbike", "motorcycle")
        lanes.append({
            "id": i + 1,
            "name": f"L{i+1}",
            "type": lane_type,
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
    warped = cv2.perspectiveTransform(pts, H.astype(np.float32)).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in warped]

def point_in_poly(px: int, py: int, poly: List[Tuple[int, int]]) -> bool:
    contour = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0

def assign_lane_bottom_center(bbox, lanes_px) -> Optional[int]:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int(y2) - 1
    for lane in lanes_px:
        if point_in_poly(cx, cy, lane["poly"]):
            return lane["id"]
    return None

# =========================
# Plate mask (near only)
# =========================
def should_mask(bbox_xyxy, img_h: int) -> bool:
    if not MASK_PLATE:
        return False
    x1, y1, x2, y2 = bbox_xyxy
    bh = max(0.0, y2 - y1)
    near_by_y = (y2 >= img_h * MASK_NEAR_Y2_NORM)
    near_by_size = (bh >= MASK_NEAR_MIN_H)
    if not MASK_NEAR_ONLY:
        return True
    return bool(near_by_y or near_by_size)

def mask_plate(img, bbox):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, x1)); y1 = int(max(0, y1))
    x2 = int(min(img.shape[1]-1, x2)); y2 = int(min(img.shape[0]-1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # xe xa / bbox nhỏ => bỏ qua
    if bh < 80:
        return

    # vùng biển số nhỏ hơn để tránh “khối đen to”
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
# Reference-based Warp (NO DRIFT)
# =========================
class ReferenceAffineWarp:
    """
    Warp theo keyframe cố định (reference) => không bị drift khi camera đứng yên.
    ORB + RANSAC trên ảnh nhỏ + deadband + EMA.
    """
    def __init__(self, ref_frame_bgr, road_poly_px=None, roi_ymax=0.60, warp_scale=0.5,
                 ema=0.25, deadband_px=2.0, deadband_scale=0.003, deadband_deg=0.25,
                 mask_road=True):
        self.scale = float(warp_scale)
        self.ema = float(ema)
        self.deadband_px = float(deadband_px)
        self.deadband_scale = float(deadband_scale)
        self.deadband_deg = float(deadband_deg)

        self.det = cv2.ORB_create(nfeatures=1600, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        ref_small = cv2.resize(ref_frame_bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)

        h, w = self.ref_gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (w, int(h * roi_ymax)), 255, -1)

        if mask_road and road_poly_px is not None and len(road_poly_px) == 4:
            rp = [(int(x * self.scale), int(y * self.scale)) for (x, y) in road_poly_px]
            cv2.fillPoly(mask, [np.array(rp, np.int32).reshape(-1, 1, 2)], 0)

        self.mask = mask
        self.ref_kp, self.ref_des = self.det.detectAndCompute(self.ref_gray, self.mask)

        self.fail_count = 0
        self.last_good = True
        self.last_inlier = 0.0
        self.last_good_matches = 0

        self.A_smooth = np.array([[1, 0, 0],
                                  [0, 1, 0]], dtype=np.float32)
        self.H_last = np.eye(3, dtype=np.float32)

        s = self.scale
        self.S = np.array([[s, 0, 0],
                           [0, s, 0],
                           [0, 0, 1]], dtype=np.float32)
        self.S_inv = np.array([[1.0/s, 0, 0],
                               [0, 1.0/s, 0],
                               [0, 0, 1]], dtype=np.float32)

    @staticmethod
    def _affine_to_H(A2x3: np.ndarray) -> np.ndarray:
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = A2x3
        return H

    @staticmethod
    def _motion_stats(A2x3: np.ndarray):
        a00, a01, tx = float(A2x3[0, 0]), float(A2x3[0, 1]), float(A2x3[0, 2])
        a10, a11, ty = float(A2x3[1, 0]), float(A2x3[1, 1]), float(A2x3[1, 2])
        scale = (a00*a00 + a10*a10) ** 0.5
        rot = np.degrees(np.arctan2(a10, a00))
        return tx, ty, scale, rot

    def estimate(self, frame_bgr):
        frm_small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frm_small, cv2.COLOR_BGR2GRAY)

        kp, des = self.det.detectAndCompute(gray, self.mask)
        if des is None or self.ref_des is None or len(kp) < 30 or len(self.ref_kp) < 30:
            self.fail_count += 1
            self.last_good = False
            return self.H_last, False

        matches = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        self.last_good_matches = len(good)
        if len(good) < MIN_GOOD_MATCHES:
            self.fail_count += 1
            self.last_good = False
            return self.H_last, False

        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        A_small, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ,
            maxIters=800, confidence=0.99
        )
        if A_small is None or inliers is None:
            self.fail_count += 1
            self.last_good = False
            return self.H_last, False

        inlier_ratio = float(inliers.mean())
        self.last_inlier = inlier_ratio
        if inlier_ratio < MIN_INLIER_RATIO:
            self.fail_count += 1
            self.last_good = False
            return self.H_last, False

        H_small = self._affine_to_H(A_small.astype(np.float32))
        H_full = self.S_inv @ H_small @ self.S
        A_full = H_full[:2, :].astype(np.float32)

        tx, ty, sc, rot = self._motion_stats(A_full)
        if (abs(tx) < self.deadband_px and abs(ty) < self.deadband_px and
            abs(sc - 1.0) < self.deadband_scale and abs(rot) < self.deadband_deg):
            A_full = np.array([[1, 0, 0],
                               [0, 1, 0]], dtype=np.float32)

        self.A_smooth = (1.0 - self.ema) * self.A_smooth + self.ema * A_full
        self.H_last = self._affine_to_H(self.A_smooth)

        self.fail_count = 0
        self.last_good = True
        return self.H_last, True

# =========================
# EMA Box smoother
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
    if SAVE_VIOLATIONS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts", "frame", "track_id", "class", "lane_id", "status"])

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

    warp = ReferenceAffineWarp(
        f0,
        road_poly_px=road_ref_px,
        roi_ymax=ROI_YMAX,
        warp_scale=WARP_SCALE,
        ema=WARP_EMA,
        deadband_px=DEADBAND_PX,
        deadband_scale=DEADBAND_SCALE,
        deadband_deg=DEADBAND_DEG,
        mask_road=MASK_ROAD_IN_WARP
    ) if USE_WARP else None

    smoother = BoxSmoother(alpha=SMOOTH_ALPHA)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(YOLO_WEIGHTS)
    device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"

    cv2.namedWindow("ITS Stream", cv2.WINDOW_NORMAL)

    frame_idx = 0
    fps_ema = 0.0
    fps_a = 0.08

    # buffered logging (đỡ lag)
    log_buf = []
    LOG_FLUSH_EVERY = 30

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if WORK_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)

        frame_idx += 1
        out = frame.copy()

        # ---- Warp update (reference-based => no drift)
        H = np.eye(3, dtype=np.float32)
        warp_ok = True
        if USE_WARP and warp is not None:
            if (frame_idx % WARP_UPDATE_EVERY) == 0:
                H, warp_ok = warp.estimate(frame)
            else:
                H = warp.H_last
                warp_ok = warp.last_good

            if warp.fail_count >= REBASE_FAILS:
                road_cur = warp_points(road_ref_px, H)
                road_ref_px = road_cur
                lanes_ref_px = generate_lane_polys_pixel(road_ref_px, BOUNDARY_TS, LANE_TYPES)
                allowed_lanes_by_cls = build_allowed_lanes(lanes_ref_px)
                warp = ReferenceAffineWarp(
                    frame,
                    road_poly_px=road_ref_px,
                    roi_ymax=ROI_YMAX,
                    warp_scale=WARP_SCALE,
                    ema=WARP_EMA,
                    deadband_px=DEADBAND_PX,
                    deadband_scale=DEADBAND_SCALE,
                    deadband_deg=DEADBAND_DEG,
                    mask_road=MASK_ROAD_IN_WARP
                )
                H = np.eye(3, dtype=np.float32)
                warp_ok = True

        road_cur_px = warp_points(road_ref_px, H)
        lanes_cur_px = []
        for ln in lanes_ref_px:
            lanes_cur_px.append({
                "id": ln["id"],
                "name": ln["name"],
                "type": ln["type"],
                "poly": warp_points(ln["poly"], H),
            })

        # ---- draw road/lanes
        cv2.polylines(out, [np.array(road_cur_px, np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2)
        for lane in lanes_cur_px:
            color = (255, 0, 0) if lane["type"] == "car" else (0, 255, 255)
            cv2.polylines(out, [np.array(lane["poly"], np.int32).reshape(-1, 1, 2)], True, color, 2)

        # ---- YOLO + ByteTrack
        res = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_TH,
            classes=TARGET_CLASS_IDS,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            device=device,
            verbose=False
        )[0]

        boxes = res.boxes
        ids = None if boxes.id is None else boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((0,), dtype=int)

        violations = 0

        for i in range(len(xyxy)):
            cls_id = int(cls[i])
            name = model.names[cls_id] if isinstance(model.names, (list, tuple)) else model.names.get(cls_id, str(cls_id))
            cls_name = "motorcycle" if name in ["motorbike", "motorcycle"] else name

            tid = int(ids[i]) if ids is not None else (i + 1)

            bbox_sm = smoother.update(tid, xyxy[i])
            x1, y1, x2, y2 = [float(v) for v in bbox_sm]

            lane_id = assign_lane_bottom_center((x1, y1, x2, y2), lanes_cur_px)
            allowed = allowed_lanes_by_cls.get(cls_name, set())
            is_viol = (lane_id is not None) and (len(allowed) > 0) and (lane_id not in allowed)

            if is_viol:
                violations += 1
                if SAVE_VIOLATIONS:
                    log_buf.append([time.time(), frame_idx, tid, cls_name, lane_id, "VIOLATION"])

            # mask near only
            if should_mask((x1, y1, x2, y2), img_h=Hh):
                mask_plate(out, (x1, y1, x2, y2))

            # draw bbox
            color = (0, 0, 255) if is_viol else (0, 255, 0)
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            ix1 = max(0, min(Ww - 1, ix1))
            iy1 = max(0, min(Hh - 1, iy1))
            ix2 = max(0, min(Ww - 1, ix2))
            iy2 = max(0, min(Hh - 1, iy2))
            if ix2 > ix1 and iy2 > iy1:
                cv2.rectangle(out, (ix1, iy1), (ix2, iy2), color, 2)

        # flush log buffer
        if SAVE_VIOLATIONS and len(log_buf) >= LOG_FLUSH_EVERY:
            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(log_buf)
            log_buf.clear()

        # FPS ema
        dt = max(1e-6, time.time() - t0)
        fps_now = 1.0 / dt
        fps_ema = fps_now if fps_ema <= 0 else (1 - fps_a) * fps_ema + fps_a * fps_now

        # HUD
        inl = warp.last_inlier if (USE_WARP and warp is not None) else 0.0
        goodm = warp.last_good_matches if (USE_WARP and warp is not None) else 0
        cv2.putText(out,
                    f"fps={fps_ema:.1f}  veh={len(xyxy)}  viol={violations}  detScale={WORK_SCALE:.2f} imgsz={IMGSZ}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out,
                    f"warp={'OK' if warp_ok else 'HOLD'} inl={inl:.2f} good={goodm} every={WARP_UPDATE_EVERY} nearMask={MASK_NEAR_ONLY}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # display
        disp = out
        if DISPLAY_SCALE != 1.0:
            disp = cv2.resize(out, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("ITS Stream", disp)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    # flush remaining log
    if SAVE_VIOLATIONS and len(log_buf) > 0:
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(log_buf)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
