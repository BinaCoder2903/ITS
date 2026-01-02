import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data","video", "test 3.mp4")

ROAD_POLY_NORM = [
    (0.1224, 0.9111),
    (0.9891, 0.9046),
    (0.7052, 0.4157),
    (0.3443, 0.4343),
]

NUM_LANES = 3  # số lane bạn đang dùng

points_clicked = []
frame = None
lb = None
rb = None
lt = None
rt = None


def proj_t_on_bottom_edge(px, py):
    """
    Chiếu điểm P xuống đoạn [LB, RB] để lấy tham số t trong [0,1]
    """
    lb_arr = np.array(lb, dtype=np.float32)
    rb_arr = np.array(rb, dtype=np.float32)
    p_arr = np.array([px, py], dtype=np.float32)

    v = rb_arr - lb_arr
    w = p_arr - lb_arr
    denom = float(v.dot(v)) + 1e-8
    t = float(v.dot(w) / denom)
    t = max(0.0, min(1.0, t))
    return t


def mouse_callback(event, x, y, flags, param):
    global points_clicked, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_clicked) >= NUM_LANES - 1:
            return

        t = proj_t_on_bottom_edge(x, y)
        points_clicked.append((x, y, t))

        print(f"[CLICK] P{len(points_clicked)} = ({x}, {y}) -> t = {t:.4f}")

        # Vẽ đường biên lane lên frame cho dễ nhìn
        b_left = lerp(lb, rb, t)
        t_left_pt = lerp(lt, rt, t)
        cv2.line(
            frame,
            (int(b_left[0]), int(b_left[1])),
            (int(t_left_pt[0]), int(t_left_pt[1])),
            (0, 0, 255),
            2,
        )

        cv2.imshow("Pick Lane Boundaries", frame)

        if len(points_clicked) == NUM_LANES - 1:
            ts_inner = [pt[2] for pt in points_clicked]
            ts_inner.sort()
            boundary_ts = [0.0] + ts_inner + [1.0]

            print("\n=== BOUNDARY_TS (dán vào detect_lanes_roadpoly.py) ===")
            print("BOUNDARY_TS = [")
            for t in boundary_ts:
                print(f"    {t:.4f},")
            print("]")
            print("Thứ tự: 0.0, biên L1-2, biên L2-3, ..., 1.0")

            cv2.waitKey(0)
            cv2.destroyAllWindows()


def lerp(p1, p2, t: float):
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)


def main():
    global frame, lb, rb, lt, rt

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Không mở được video:", VIDEO_SOURCE)
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Không đọc được frame đầu tiên.")
        return

    h, w = frame.shape[:2]
    # denorm road poly
    pts = [(int(x * w), int(y * h)) for x, y in ROAD_POLY_NORM]
    lb, rb, rt, lt = pts[0], pts[1], pts[2], pts[3]

    # vẽ road polygon
    cv2.polylines(
        frame,
        [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
        True,
        (255, 255, 0),
        2,
    )

    cv2.namedWindow("Pick Lane Boundaries", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pick Lane Boundaries", mouse_callback)

    print("Hướng dẫn chọn biên lane:")
    print(f"- Video: {VIDEO_SOURCE}")
    print("- Đã có ROAD_POLY_NORM (4 góc mặt đường).")
    print(f"- NUM_LANES = {NUM_LANES} -> cần click {NUM_LANES-1} lần:")
    print("   1) Click vào VẠCH GIỮA LANE 1-2 (gần phía dưới).")
    print("   2) Click vào VẠCH GIỮA LANE 2-3 (gần phía dưới).")
    print("   ...")
    print("Sau khi đủ, script sẽ in ra BOUNDARY_TS.")

    cv2.imshow("Pick Lane Boundaries", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
