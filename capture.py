import cv2 as cv
import os
import glob

SAVE_DIR = "./images"

# =========================
# 1. 폴더 초기화
# =========================
if os.path.exists(SAVE_DIR):
    files = glob.glob(os.path.join(SAVE_DIR, "*"))
    for f in files:
        os.remove(f)
else:
    os.makedirs(SAVE_DIR)

print("[INFO] images 폴더 초기화 완료")

# =========================
# 2. 카메라 실행
# =========================
cap = cv.VideoCapture(0)
assert cap.isOpened(), "카메라 열기 실패"

print("[INFO] 촬영 시작")
print(" - 's': 촬영")
print(" - 'q': 종료")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 화면 표시
    display = frame.copy()
    cv.putText(display, f"Captured: {count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv.imshow("Capture", display)

    key = cv.waitKey(1) & 0xFF

    # =========================
    # 수동 촬영
    # =========================
    if key == ord('s'):
        filename = os.path.join(SAVE_DIR, f"img_{count:03d}.jpg")
        cv.imwrite(filename, frame)
        print(f"[SAVED] {filename}")
        count += 1

    # =========================
    # 종료
    # =========================
    elif key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()