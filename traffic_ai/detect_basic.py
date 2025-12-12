from ultralytics import YOLO
import cv2
import os

# ---- CẤU HÌNH NGUỒN VIDEO ----
# Dùng đường dẫn tương đối theo vị trí file detect_basic.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa detect_basic.py
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video", "test.mp4")

# Tỉ lệ thu nhỏ khi hiển thị (0.5 = 50%, 0.7 = 70%, ...)
DISPLAY_SCALE = 0.6

print("[DEBUG] VIDEO_SOURCE =", VIDEO_SOURCE)
print("[DEBUG] File tồn tại? :", os.path.exists(VIDEO_SOURCE))


def main():
    # ---- LOAD MODEL YOLOV8 ----
    print("[INFO] Đang load model YOLOv8...")
    model = YOLO("yolov8n.pt")  # lần đầu sẽ tự tải về

    # ---- MỞ VIDEO / WEBCAM ----
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được nguồn video: {VIDEO_SOURCE}")
        return

    # Các class giao thông quan tâm (tên theo COCO)
    target_classes = {"car", "motorbike", "bus", "truck"}

    # Cho phép cửa sổ có thể thay đổi kích thước
    cv2.namedWindow("YOLOv8 Traffic Demo", cv2.WINDOW_NORMAL)

    print("[INFO] Bắt đầu xử lý. Nhấn phím 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Hết video hoặc không đọc được frame.")
            break

        # ---- CHẠY YOLO TRÊN FRAME ----
        results = model(frame, conf=0.4)[0]

        # ---- VẼ BOUNDING BOX CHO XE ----
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Bỏ qua đối tượng không phải xe
            if cls_name not in target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Vẽ khung
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
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

        # ---- THU NHỎ FRAME TRƯỚC KHI HIỂN THỊ ----
        h, w = frame.shape[:2]
        new_w, new_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
        frame_display = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # ---- HIỂN THỊ KẾT QUẢ ----
        cv2.imshow("YOLOv8 Traffic Demo", frame_display)

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát.")


if __name__ == "__main__":
    main()
