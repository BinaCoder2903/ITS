import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data","video", "test 3.mp4")

points = []  # sẽ lưu 4 điểm (x, y)
frame = None

def mouse_callback(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"[CLICK] ({x}, {y})")
        # vẽ chấm lên ảnh cho dễ nhìn
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Pick Road Polygon", frame)

        if len(points) == 4:
            h, w = frame.shape[:2]
            norm = [(px / w, py / h) for (px, py) in points]
            print("\n=== 4 điểm pixel ===")
            for i, (px, py) in enumerate(points, start=1):
                print(f"P{i}: ({px}, {py})")
            print("\n=== ROAD_POLY_NORM (dán vào code) ===")
            print("[")
            for (nx, ny) in norm:
                print(f"    ({nx:.4f}, {ny:.4f}),")
            print("]")
            print("\nThứ tự điểm: left_bottom, right_bottom, right_top, left_top")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    global frame
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Không mở được video:", VIDEO_SOURCE)
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Không đọc được frame đầu tiên.")
        return

    cv2.namedWindow("Pick Road Polygon", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pick Road Polygon", mouse_callback)

    print("Hướng dẫn:")
    print(" 1️⃣ Click lần lượt 4 điểm mép ĐƯỜNG theo thứ tự:")
    print("     1. left_bottom  = mép đường trái, gần camera (gần chân ảnh)")
    print("     2. right_bottom = mép đường phải, gần camera")
    print("     3. right_top    = mép đường phải, xa camera")
    print("     4. left_top     = mép đường trái, xa camera")
    print(" 2️⃣ Sau khi đủ 4 điểm, script sẽ in ROAD_POLY_NORM để bạn copy.")
    print()

    cv2.imshow("Pick Road Polygon", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
