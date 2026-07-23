import cv2
import time
from ultralytics import YOLO

def main():
    model = YOLO("best.pt")

   
    print("1. Live Camera Mode")
    print("2. Video File Mode")
    choice = input("Select Mode (1 or 2): ").strip()

    target_class = input("Enter object class name to count (e.g., apple) or press Enter to skip: ").strip().lower()

    save_video = False
    out = None

    if choice == "1":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access webcam.")
            return
    elif choice == "2":
        video_path = input("Enter offline video file path: ").strip()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'.")
            return

        save_video = True
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Fallback FPS

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))
    else:
        print("Invalid choice selection.")
        return

    prev_time = 0
    print("\n[INFO] Starting detection. Press 'q' on the video window to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        target_count = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                # Count target object
                if target_class and cls_name.lower() == target_class:
                    target_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{cls_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Calc FPS
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display FPS
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display Object Count
        if target_class:
            cv2.putText(frame, f"Count ({target_class}): {target_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if save_video and out is not None:
            out.write(frame)

        cv2.imshow("YOLO Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if save_video and out is not None:
        out.release()
        print("[INFO] Annotated video saved successfully as 'annotated_output.mp4'.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()