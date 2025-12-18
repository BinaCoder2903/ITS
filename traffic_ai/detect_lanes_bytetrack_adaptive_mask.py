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

VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 2.mp4")
YOLO_WEIGHTS = "yolov8n.pt"
TRACKER_CFG = "bytetrack.yaml"

# Speed
WORK_SCALE = 0.85
DISPLAY_SCALE = 0.85
IMGSZ = 480
CONF_TH = 0.35
MAX_DET = 60
TARGET_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)

# Smooth bbox
SMOOTH_ALPHA = 0.65

# Warp stabilize
USE_WARP = True
WARP_UPDATE_EVERY = 10
ROI_YMAX = 0.65                 # lấy feature ở phần trên
SIDE_MARGIN_NORM = 0.16         # ưu tiên lề trái/phải
MIN_GOOD_MATCHES = 80
MIN_INLIER_RATIO = 0.25
RANSAC_REPROJ = 4.0
REBASE_FAILS = 25

# Deadband + smoothing cho warp (giảm drift khi cam đứng yên)
DEAD_TRANS_PX = 1.2             # tịnh tiến nhỏ hơn ngưỡng => coi như 0
DEAD_SCALE = 0.004              # |scale-1| nhỏ => coi như 0
H_SMOOTH_GAMMA = 0.20           # 0..1, nhỏ hơn = mượt hơn nhưng trễ hơn

# Mask biển số
MASK_PLATE = True
MASK_MODE = "fill"              # "fill" nhanh; "blur" nặng hơn

MASK_NEAR_ONLY = True
MASK_NEAR_Y2_NORM = 0.72        # chỉ mask nếu đáy bbox nằm dưới 72% chiều cao
MASK_NEAR_MIN_H = 120           # hoặc bbox cao >= 120px

# Output log
SAVE_VIOLATIONS = True
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_CSV = os.path.join(OUTPUT_DIR, "violations_log.csv")

try:
    cv2.setNumThreads(1)
except Exception:
    pass

# =========================
# ROAD/LANE (NORM 0..1)
# order: left_bottom, right_bottom, right_top, left_top
# =========================
ROAD_POLY_NORM = [
    (0.0719, 0.9454),
    (0.5984, 0.9704),
    (0.5927, 0.5370),
    (0.4484, 0.5315),
]


BOUNDARY_TS = [
    0.0000,
    0.2613,
    0.6295,
    1.0000,
]
LANE_TYPES = ["car", "car", "car"]  # ví dụ: ["car","car","motorcycle"]

# =========================
# Geometry
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

# =========================
# Mask plate (near-only)
# =========================
def should_mask(bbox_xyxy, img_h: int) -> bool:
    if not MASK_PLATE:
        return False
    x1, y1, x2, y2 = bbox_xyxy
    bh = max(0.0, y2 - y1)
    if not MASK_NEAR_ONLY:
        return True
    near_by_y = (y2 >= img_h * MASK_NEAR_Y2_NORM)
    near_by_size = (bh >= MASK_NEAR_MIN_H)
    return bool(near_by_y or near_by_size)

def mask_plate(img, bbox):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, x1)); y1 = int(max(0, y1))
    x2 = int(min(img.shape[1]-1, x2)); y2 = int(min(img.shape[0]-1, y2))
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
# Warp (AKAZE + RANSAC) + static mask + deadband + smoothing
# =========================
def affine_to_H(A2x3: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = A2x3.astype(np.float32)
    return H

def H_to_affine(H: np.ndarray) -> np.ndarray:
    return H[:2, :].astype(np.float32)

def affine_scale(A2x3: np.ndarray) -> float:
    a, b, _ = A2x3[0]
    c, d, _ = A2x3[1]
    s1 = np.sqrt(a*a + b*b)
    s2 = np.sqrt(c*c + d*d)
    return float((s1 + s2) * 0.5)

def affine_trans(A2x3: np.ndarray) -> float:
    tx = float(A2x3[0, 2])
    ty = float(A2x3[1, 2])
    return float(np.sqrt(tx*tx + ty*ty))

class IncrementalAffineWarp:
    def __init__(self, ref_frame_bgr, road_poly_px: List[Tuple[int, int]]):
        self.det = cv2.AKAZE_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.H_total = np.eye(3, dtype=np.float32)
        self.A_total = np.eye(2, 3, dtype=np.float32)

        self.prev_gray = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = self.prev_gray.shape[:2]

        # mask: ưu tiên TOP + hai lề; loại vùng mặt đường để tránh xe
        mask = np.zeros((h, w), dtype=np.uint8)

        top_h = int(h * ROI_YMAX)
        cv2.rectangle(mask, (0, 0), (w, top_h), 255, -1)

        side = int(w * SIDE_MARGIN_NORM)
        cv2.rectangle(mask, (0, 0), (side, h), 255, -1)
        cv2.rectangle(mask, (w - side, 0), (w, h), 255, -1)

        # loại road polygon (mặt đường/xe chạy) khỏi mask
        road_cnt = np.array(road_poly_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [road_cnt], 0)

        self.mask = mask

        self.prev_kp, self.prev_des = self.det.detectAndCompute(self.prev_gray, self.mask)

        self.fail_count = 0
        self.last_good = True
        self.last_inlier = 0.0
        self.last_good_matches = 0
        self.last_scale = 1.0
        self.last_trans = 0.0

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.det.detectAndCompute(gray, self.mask)

        if des is None or self.prev_des is None or len(kp) < 80 or len(self.prev_kp) < 80:
            self.fail_count += 1
            self.last_good = False
            return self.H_total, False

        matches = self.matcher.knnMatch(self.prev_des, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        self.last_good_matches = len(good)
        if len(good) < MIN_GOOD_MATCHES:
            self.fail_count += 1
            self.last_good = False
            return self.H_total, False

        src = np.float32([self.prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        A, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ,
            maxIters=1500, confidence=0.99
        )
        if A is None or inliers is None:
            self.fail_count += 1
            self.last_good = False
            return self.H_total, False

        inlier_ratio = float(inliers.mean())
        self.last_inlier = inlier_ratio
        if inlier_ratio < MIN_INLIER_RATIO:
            self.fail_count += 1
            self.last_good = False
            return self.H_total, False

        A = A.astype(np.float32)
        sc = affine_scale(A)
        tr = affine_trans(A)
        self.last_scale = sc
        self.last_trans = tr

        # DEAD-BAND: camera gần như đứng yên => bỏ update để khỏi drift
        if (tr < DEAD_TRANS_PX) and (abs(sc - 1.0) < DEAD_SCALE):
            # vẫn cập nhật prev để match ổn định, nhưng giữ H_total
            self.prev_gray = gray
            self.prev_kp, self.prev_des = kp, des
            self.fail_count = 0
            self.last_good = True
            return self.H_total, True

        # update tổng + smooth
        A_new = (A @ np.vstack([self.A_total, [0, 0, 1]]))[:2, :]  # compose (affine)
        self.A_total = (1.0 - H_SMOOTH_GAMMA) * self.A_total + H_SMOOTH_GAMMA * A_new
        self.H_total = affine_to_H(self.A_total)

        self.prev_gray = gray
        self.prev_kp, self.prev_des = kp, des
        self.fail_count = 0
        self.last_good = True
        return self.H_total, True

# =========================
# EMA smoother for bbox
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

    warp = IncrementalAffineWarp(f0, road_ref_px) if USE_WARP else None
    smoother = BoxSmoother(alpha=SMOOTH_ALPHA)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(YOLO_WEIGHTS)
    cv2.namedWindow("ITS Stream", cv2.WINDOW_NORMAL)

    frame_idx = 0
    fps = 0.0
    fps_t = time.time()

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if WORK_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)

        frame_idx += 1
        out = frame.copy()

        # Warp update thưa
        H = np.eye(3, dtype=np.float32)
        warp_ok = True
        if USE_WARP and warp is not None:
            if (frame_idx % WARP_UPDATE_EVERY) == 0:
                H, warp_ok = warp.update(frame)
            else:
                H = warp.H_total
                warp_ok = warp.last_good

            if warp.fail_count >= REBASE_FAILS:
                # rebase lại theo frame hiện tại để tránh drift dài hạn
                road_ref_px = warp_points(road_ref_px, H)
                lanes_ref_px = generate_lane_polys_pixel(road_ref_px, BOUNDARY_TS, LANE_TYPES)
                allowed_lanes_by_cls = build_allowed_lanes(lanes_ref_px)
                warp = IncrementalAffineWarp(frame, road_ref_px)
                H = np.eye(3, dtype=np.float32)

        road_cur_px = warp_points(road_ref_px, H)
        lanes_cur_px = [{
            "id": ln["id"],
            "name": ln["name"],
            "type": ln["type"],
            "poly": warp_points(ln["poly"], H),
        } for ln in lanes_ref_px]

        cv2.polylines(out, [np.array(road_cur_px, np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2)
        for lane in lanes_cur_px:
            color = (255, 0, 0) if lane["type"] == "car" else (0, 255, 255)
            cv2.polylines(out, [np.array(lane["poly"], np.int32).reshape(-1, 1, 2)], True, color, 2)

        # YOLO + ByteTrack
        res = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_TH,
            classes=TARGET_CLASS_IDS,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            verbose=False
        )[0]

        boxes = res.boxes
        ids = None if boxes.id is None else boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((0,), dtype=int)

        violations = 0

        for i in range(len(xyxy)):
            cls_id = int(cls[i])
            cls_name = model.names.get(cls_id, str(cls_id))
            if cls_name in ["motorbike", "motorcycle"]:
                cls_name = "motorcycle"

            tid = int(ids[i]) if ids is not None else (i + 1)

            bbox_sm = smoother.update(tid, xyxy[i])
            x1, y1, x2, y2 = [float(v) for v in bbox_sm]

            lane_id = assign_lane_bottom_center((x1, y1, x2, y2), lanes_cur_px)
            allowed = allowed_lanes_by_cls.get(cls_name, set())
            is_viol = (lane_id is not None) and (len(allowed) > 0) and (lane_id not in allowed)
            if is_viol:
                violations += 1

            if should_mask((x1, y1, x2, y2), img_h=Hh):
                mask_plate(out, (x1, y1, x2, y2))

            color = (0, 0, 255) if is_viol else (0, 255, 0)
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            ix1 = max(0, min(Ww - 1, ix1))
            iy1 = max(0, min(Hh - 1, iy1))
            ix2 = max(0, min(Ww - 1, ix2))
            iy2 = max(0, min(Hh - 1, iy2))
            if ix2 > ix1 and iy2 > iy1:
                cv2.rectangle(out, (ix1, iy1), (ix2, iy2), color, 2)

            if SAVE_VIOLATIONS and is_viol:
                with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([time.time(), frame_idx, tid, cls_name, lane_id, "VIOLATION"])

        now = time.time()
        dt = max(1e-6, now - t0)
        if now - fps_t >= 0.4:
            fps = 1.0 / dt
            fps_t = now

        inl = warp.last_inlier if (USE_WARP and warp is not None) else 0.0
        goodm = warp.last_good_matches if (USE_WARP and warp is not None) else 0
        sc = warp.last_scale if (USE_WARP and warp is not None) else 1.0
        tr = warp.last_trans if (USE_WARP and warp is not None) else 0.0

        cv2.putText(out,
                    f"fps={fps:.1f} veh={len(xyxy)} viol={violations} detScale={WORK_SCALE} imgsz={IMGSZ}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out,
                    f"warp={'OK' if warp_ok else 'HOLD'} inl={inl:.2f} good={goodm} every={WARP_UPDATE_EVERY} s={sc:.3f} t={tr:.1f} nearMask={MASK_NEAR_ONLY}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        disp = out
        if DISPLAY_SCALE != 1.0:
            disp = cv2.resize(out, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("ITS Stream", disp)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
