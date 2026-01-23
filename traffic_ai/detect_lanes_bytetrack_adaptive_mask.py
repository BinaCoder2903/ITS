from ultralytics import YOLO
import cv2, os, time, csv
import numpy as np

# --- CẤU HÌNH TỐI ƯU HÓA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test 3.mp4")

# 1. Dùng model Nano để nhanh hơn gấp 3 lần
YOLO_WEIGHTS = "yolov8n.pt" 

# 2. Tắt chống rung nếu camera cố định (đặt False sẽ tăng FPS đáng kể)
ENABLE_STABILIZATION = True 

# 3. Giảm độ phân giải ảnh khi đưa vào AI (640 là chuẩn, thấp hơn sẽ mờ)
IMGSZ = 640 

# 4. Giảm kích thước khung hình xử lý (0.7 = 70% kích thước gốc)
# Tăng lên 0.7 để hình rõ hơn, giúp AI bắt được xe
WORK_SCALE = 0.70 
DISPLAY_SCALE = 1.0 # Tỉ lệ hiển thị màn hình

# 5. Bỏ qua frame (2 = xử lý 1 frame, bỏ 1 frame)
VID_STRIDE = 2

CPU_THREADS = max(1, (os.cpu_count() or 8) - 1)
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)

MAX_DET = 100 # Giảm số lượng detect tối đa
CONF_TRACK = 0.15 # Giảm ngưỡng tự tin xuống thấp (0.15) để bắt xe nhạy hơn
TARGET_CLASS_IDS = [2, 3, 5, 7] # car, motorcycle, bus, truck

# --- CẤU HÌNH NGHIỆP VỤ (GIỮ NGUYÊN) ---
CAR_MIN_CONF = 0.30
MOTO_MIN_CONF = 0.25
BUS_MIN_CONF = 0.30
TRUCK_MIN_CONF = 0.30

ROAD_POLY_NORM = [
    (0.1208, 0.9176),
    (0.9938, 0.9157),
    (0.6885, 0.3519),
    (0.3672, 0.3556),
]
BOUNDARY_TS = [0.0000, 0.2194, 0.4742, 0.7374, 1.0000]
LANE_TYPES = ["motorcycle", "motorcycle", "car", "car"]

# Cấu hình chống rung tối ưu (ORB thay vì AKAZE)
WARP_UPDATE_EVERY = 30 # Cập nhật ma trận biến đổi ít thường xuyên hơn
WARP_SCALE = 0.40      # Thu nhỏ ảnh hơn nữa khi tính toán feature
ROI_YMAX = 0.60
MIN_GOOD_MATCHES = 15  # Giảm yêu cầu số điểm khớp cho ORB
MIN_INLIER_RATIO = 0.15
RANSAC_REPROJ = 5.0
SMOOTH_GAMMA = 0.20 # Giảm độ mượt để phản ứng nhanh hơn
DEAD_TRANS_PX = 5.0 # Tăng vùng chết để tránh rung lắc nhỏ
DEAD_SCALE = 0.005

# TẮT ROI ĐỂ TRÁNH CẮT MẤT XE
DETECT_ROI = False
ROI_PAD_X = 0.05
ROI_PAD_Y = 0.20

MASK_PLATE = True
MASK_NEAR_Y2_NORM = 0.75
MASK_NEAR_MIN_H = 100

SAVE_VIOLATIONS = True
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_CSV = os.path.join(OUTPUT_DIR, "violations_log.csv")
DRAW_TEXT = True # Bật text để dễ debug

# Tối ưu hóa OpenCV
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(CPU_THREADS)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(CPU_THREADS)
except Exception:
    torch = None

def ensure_tracker_cfg(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bytetrack_fast.yaml") # Đổi tên file config
    if os.path.exists(path):
        return path
    
    # Tinh chỉnh Tracker cho tốc độ (match_thresh thấp hơn, buffer nhỏ hơn)
    yml = """tracker_type: bytetrack
track_high_thresh: 0.4
track_low_thresh: 0.1
new_track_thresh: 0.3
track_buffer: 30
match_thresh: 0.8
fuse_score: True
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(yml)
    return path

# --- CÁC HÀM TIỆN ÍCH GIỮ NGUYÊN ---
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
        bL = lerp(lb, rb, tL); bR = lerp(lb, rb, tR)
        tR_pt = lerp(lt, rt, tR); tL_pt = lerp(lt, rt, tL)
        lanes.append({
            "id": i + 1,
            "type": lane_types[i].replace("motorbike", "motorcycle"),
            "poly": [(int(bL[0]), int(bL[1])), (int(bR[0]), int(bR[1])),
                     (int(tR_pt[0]), int(tR_pt[1])), (int(tL_pt[0]), int(tL_pt[1]))]
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
    if not MASK_PLATE or cls_name == "motorcycle": return False
    x1, y1, x2, y2 = bbox_xyxy
    bh = max(0.0, y2 - y1)
    return (y2 >= img_h * MASK_NEAR_Y2_NORM) or (bh >= MASK_NEAR_MIN_H)

def mask_plate(img, bbox):
    x1, y1, x2, y2 = bbox
    # Logic mask đơn giản hóa
    bw, bh = x2 - x1, y2 - y1
    if bh < 50: return # Bỏ qua xe quá nhỏ
    
    px1, px2 = int(x1 + 0.3 * bw), int(x1 + 0.7 * bw)
    py1, py2 = int(y1 + 0.75 * bh), int(y2)
    
    # Kẹp giá trị trong khung hình
    h_img, w_img = img.shape[:2]
    px1 = max(0, px1); py1 = max(0, py1)
    px2 = min(w_img, px2); py2 = min(h_img, py2)
    
    if px2 > px1 and py2 > py1:
        # Dùng Mosaic (pixelate) thay vì vẽ đen để trông tự nhiên hơn (nhưng chậm hơn xíu, ở đây vẽ đen cho nhanh)
        cv2.rectangle(img, (px1, py1), (px2, py2), (20, 20, 20), -1)

def roi_from_poly(poly_px, w, h, pad_x=0.06, pad_y=0.45):
    xs = [p[0] for p in poly_px]
    ys = [p[1] for p in poly_px]
    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
    dx = int((x2 - x1) * pad_x)
    dy = int((y2 - y1) * pad_y)
    x1 = max(0, x1 - dx); x2 = min(w - 1, x2 + dx)
    y1 = max(0, y1 - dy); y2 = min(h - 1, y2 + dy)
    if x2 <= x1 or y2 <= y1: return 0, 0, w, h
    return x1, y1, x2, y2

def homography_motion_gate(H, w, h):
    # Rút gọn tính toán ma trận
    Hn = H / (H[2, 2] + 1e-8)
    # Chỉ kiểm tra tâm ảnh
    cx, cy = w * 0.5, h * 0.5
    vec = np.array([cx, cy, 1.0])
    vec_new = Hn @ vec
    nx, ny = vec_new[0]/vec_new[2], vec_new[1]/vec_new[2]
    trans = np.sqrt((nx-cx)**2 + (ny-cy)**2)
    return trans < DEAD_TRANS_PX

# --- OPTIMIZED WARPING CLASS ---
class OptimizedHomographyWarp:
    def __init__(self, ref_frame_bgr, road_poly_px):
        # SỬ DỤNG ORB THAY VÌ AKAZE (Nhanh gấp 5-10 lần)
        self.det = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # CrossCheck loại bỏ match sai tốt hơn
        
        self.last_good = True
        self.last_inlier = 0.0
        self.last_good_matches = 0
        self.H = np.eye(3, dtype=np.float32)

        ref_gray_full = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = ref_gray_full.shape[:2]

        mask_full = np.zeros((h, w), dtype=np.uint8)
        # Chỉ quan tâm vùng đường, bỏ qua vùng trời để tránh nhiễu
        cv2.fillPoly(mask_full, [np.array(road_poly_px, dtype=np.int32)], 255)
        # Mở rộng mask ra một chút
        cv2.dilate(mask_full, np.ones((15,15), np.uint8), iterations=1)

        self.ref_gray = cv2.resize(ref_gray_full, None, fx=WARP_SCALE, fy=WARP_SCALE, interpolation=cv2.INTER_NEAREST)
        self.mask = cv2.resize(mask_full, (self.ref_gray.shape[1], self.ref_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.ref_kp, self.ref_des = self.det.detectAndCompute(self.ref_gray, self.mask)

    def update(self, frame_bgr, full_w, full_h):
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_full, None, fx=WARP_SCALE, fy=WARP_SCALE, interpolation=cv2.INTER_NEAREST)

        kp, des = self.det.detectAndCompute(gray, self.mask)
        if des is None or self.ref_des is None or len(kp) < 20:
            self.last_good = False
            return self.H, False

        # ORB dùng match trực tiếp, nhanh hơn KNN
        matches = self.matcher.match(self.ref_des, des)
        # Sắp xếp theo khoảng cách
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Lấy top % tốt nhất
        num_good = int(len(matches) * 0.3)
        good = matches[:num_good]

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

        # Điều chỉnh lại tỉ lệ scale
        s = WARP_SCALE
        S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32)
        Sinv = np.array([[1/s, 0, 0], [0, 1/s, 0], [0, 0, 1]], dtype=np.float32)
        H_new = Sinv @ H_new @ S
        H_new = H_new / (H_new[2, 2] + 1e-8)

        if homography_motion_gate(H_new, full_w, full_h):
            self.last_good = True
            return self.H, True

        H_blend = (1.0 - SMOOTH_GAMMA) * self.H + SMOOTH_GAMMA * H_new
        self.H = H_blend / (H_blend[2, 2] + 1e-8)
        self.last_good = True
        return self.H, True

def pick_device():
    if torch is None: return "cpu", False
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0", True
    except: pass
    return "cpu", False

def main():
    tracker_cfg = ensure_tracker_cfg(OUTPUT_DIR)
    
    # Init CSV Log
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
        print(f"[ERROR] Cannot open: {VIDEO_SOURCE}")
        return

    ret, f0 = cap.read()
    if not ret: return

    if WORK_SCALE != 1.0:
        f0 = cv2.resize(f0, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)

    Hh, Ww = f0.shape[:2]
    road_ref_px = denorm_polygon(ROAD_POLY_NORM, Ww, Hh)
    lanes_ref_px = generate_lane_polys_pixel(road_ref_px, BOUNDARY_TS, LANE_TYPES)
    allowed_lanes_by_cls = build_allowed_lanes(lanes_ref_px)
    
    # Chỉ khởi tạo warp nếu bật tính năng
    warp = OptimizedHomographyWarp(f0, road_ref_px) if (ENABLE_STABILIZATION) else None

    print(f"[INFO] Loading model {YOLO_WEIGHTS}...")
    model = YOLO(YOLO_WEIGHTS)
    
    device, use_half = pick_device()
    print(f"[INFO] Device={device}, Half={use_half}, Input Size={IMGSZ}, Stride={VID_STRIDE}")

    cv2.namedWindow("ITS Optimized", cv2.WINDOW_NORMAL)
    
    frame_idx = 0
    fps_ema = 0.0
    last_t = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # SKIP FRAMES LOGIC
        if VID_STRIDE > 1 and (frame_idx % VID_STRIDE != 0):
            frame_idx += 1
            continue

        if WORK_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_NEAREST)

        frame_idx += 1
        out = frame.copy() # Chỉ copy nếu cần vẽ đè, có thể vẽ trực tiếp lên frame để tiết kiệm RAM

        # --- XỬ LÝ CHỐNG RUNG (NẶNG) ---
        H = np.eye(3, dtype=np.float32)
        warp_ok = True
        
        if warp is not None:
            # Chỉ update ma trận H sau mỗi N frames
            if (frame_idx % WARP_UPDATE_EVERY) == 0:
                H, warp_ok = warp.update(frame, Ww, Hh)
            else:
                H = warp.H
                warp_ok = warp.last_good

        road_cur_px = warp_points(road_ref_px, H)
        lanes_cur_px = [{"id": ln["id"], "type": ln["type"], "poly": warp_points(ln["poly"], H)} for ln in lanes_ref_px]

        # --- VẼ LÀN ---
        # Chỉ vẽ đường viền (dày 1px) cho nhẹ
        cv2.polylines(out, [np.array(road_cur_px, dtype=np.int32)], True, (255, 255, 0), 1)
        for lane in lanes_cur_px:
            col = (255, 100, 0) if lane["type"] == "car" else (0, 200, 255)
            cv2.polylines(out, [np.array(lane["poly"], np.int32)], True, col, 1)

        # --- CẮT ROI & DETECT ---
        if DETECT_ROI:
            rx1, ry1, rx2, ry2 = roi_from_poly(road_cur_px, Ww, Hh, ROI_PAD_X, ROI_PAD_Y)
            det_img = frame[ry1:ry2, rx1:rx2]
            offx, offy = rx1, ry1
        else:
            det_img = frame
            offx, offy = 0, 0

        # --- YOLO TRACKING ---
        t0 = time.perf_counter()
        
        # Tracking argument tối ưu
        results = model.track(
            det_img,
            persist=True,
            tracker=tracker_cfg,
            conf=CONF_TRACK,
            classes=TARGET_CLASS_IDS,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            verbose=False,
            device=device,
            half=use_half # Dùng FP16 nếu có GPU
        )
        
        yolo_ms = (time.perf_counter() - t0) * 1000.0
        res = results[0]
        boxes = res.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            # Offset tọa độ về khung hình gốc
            xyxy[:, [0, 2]] += offx
            xyxy[:, [1, 3]] += offy

            # --- LOGIC NGHIỆP VỤ ---
            violations = 0
            kept = 0
            
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                cls_id = clss[i]
                tid = ids[i]
                conf = confs[i]
                
                # Filter tên lớp
                cls_name = model.names[cls_id]
                if cls_name == "motorbike": cls_name = "motorcycle"
                
                if conf < class_min_conf(cls_name): continue
                kept += 1

                # Kiểm tra làn đường
                lane_id = assign_lane_bottom_center((x1, y1, x2, y2), lanes_cur_px)
                is_viol = False
                if lane_id is not None:
                    allowed = allowed_lanes_by_cls.get(cls_name, set())
                    if allowed and lane_id not in allowed:
                        is_viol = True
                        violations += 1

                # Che biển số
                if should_mask((x1, y1, x2, y2), Hh, cls_name):
                    mask_plate(out, (x1, y1, x2, y2))

                # Vẽ Box
                color = (0, 0, 255) if is_viol else (0, 255, 0)
                pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(out, pt1, pt2, color, 2)
                
                if DRAW_TEXT:
                    label = f"{tid}" # Chỉ hiện ID cho đỡ rối
                    cv2.putText(out, label, (pt1[0], max(0, pt1[1]-5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Ghi Log
                if SAVE_VIOLATIONS and is_viol and log_w:
                    log_w.writerow([time.time(), frame_idx, tid, cls_name, f"{conf:.2f}", lane_id, "VIOLATION"])

        # Tính FPS
        now = time.perf_counter()
        dt = now - last_t
        fps = 1.0 / dt if dt > 0 else 0
        fps_ema = 0.1 * fps + 0.9 * fps_ema # Smooth FPS
        last_t = now

        # Hiển thị thông số
        cv2.putText(out, f"FPS: {fps_ema:.1f} | YOLO: {yolo_ms:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if warp is not None:
            status = "OK" if warp_ok else "LOST"
            cv2.putText(out, f"Stab: {status} ({warp.last_good_matches})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

        disp = out
        if DISPLAY_SCALE != 1.0:
            disp = cv2.resize(out, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            
        cv2.imshow("ITS Optimized", disp)
        if (cv2.waitKey(1) & 0xFF) == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    if log_f: log_f.close()

if __name__ == "__main__":
    main()