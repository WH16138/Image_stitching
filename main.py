import cv2 as cv
import numpy as np
import glob

# =========================
# 1. 이미지 로드
# =========================
def load_images(path_pattern):
    paths = sorted(glob.glob(path_pattern))
    images = [cv.imread(p) for p in paths]
    assert all(img is not None for img in images), "이미지 로드 실패"
    return images


# =========================
# 2. 특징점 추출
# =========================
def extract_features(img):
    orb = cv.ORB_create(nfeatures=4000)
    kp, desc = orb.detectAndCompute(img, None)
    return kp, desc


# =========================
# 3. 매칭 (KNN + ratio + 정렬)
# =========================
def match_features(desc1, desc2):
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)

    good = good[:100]

    return good


# =========================
# 4. Homography
# =========================
def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 3.0)

    inlier_ratio = np.sum(mask) / len(mask)

    if inlier_ratio < 0.3:
        print("Homography 불안정 (inlier ratio 낮음)")
        return None

    return H

# =========================
# 5. Global Homography (핵심 수정)
# =========================
def accumulate_homographies(H_pair):
    N = len(H_pair) + 1
    H_global = [None] * N

    mid = N // 2
    H_global[mid] = np.eye(3)

    # 오른쪽 (정방향)
    for i in range(mid + 1, N):
        H_global[i] = H_global[i - 1] @ H_pair[i - 1]

    # 왼쪽 (역방향)
    for i in range(mid - 1, -1, -1):
        H_global[i] = H_global[i + 1] @ np.linalg.inv(H_pair[i])

    return H_global


# =========================
# 6. Canvas 계산
# =========================
def compute_canvas(images, Hs):
    corners_all = []

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        warped = cv.perspectiveTransform(corners[None,:,:], Hs[i])[0]
        corners_all.append(warped)

    corners_all = np.vstack(corners_all)

    x_min, y_min = np.min(corners_all, axis=0)
    x_max, y_max = np.max(corners_all, axis=0)

    return int(x_min), int(y_min), int(x_max), int(y_max)


# =========================
# 7. Crop (선택)
# =========================
def crop_black(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    coords = cv.findNonZero(thresh)
    x, y, w, h = cv.boundingRect(coords)
    return image[y:y+h, x:x+w]

def create_weight_mask(mask):
    dist = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)
    dist = dist / (dist.max() + 1e-6)
    return dist

# =========================
# 8. Stitching
# =========================
def stitch(images):
    N = len(images)

    # 특징 추출
    features = [extract_features(img) for img in images]

    # pairwise homography
    H_pair = []
    for i in range(N - 1):
        kp1, desc1 = features[i]
        kp2, desc2 = features[i + 1]

        matches = match_features(desc1, desc2)

        if len(matches) < 30:
            raise Exception(f"매칭 부족: {i}")

        H = compute_homography(kp1, kp2, matches)
        if H is None:
            raise Exception(f"Homography 실패: {i}")

        H_pair.append(H)

    # global homography
    H_global = accumulate_homographies(H_pair)

    # canvas
    x_min, y_min, x_max, y_max = compute_canvas(images, H_global)

    padding = 100
    width = int(x_max - x_min + padding)
    height = int(y_max - y_min + padding)

    T = np.array([[1, 0, -x_min + padding//2],
                  [0, 1, -y_min + padding//2],
                  [0, 0, 1]])

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for i in range(N):
        H = T @ H_global[i]
        warped = cv.warpPerspective(images[i], H, (width, height))

        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8)

        w = create_weight_mask(mask)

        canvas += warped * w[..., None]
        weight_sum += w

    weight_sum[weight_sum == 0] = 1
    result = (canvas / weight_sum[..., None]).astype(np.uint8)
    return result

# =========================
# 실행
# =========================
if __name__ == "__main__":
    images = load_images("./images/*.jpg")
    result = stitch(images)

    # 🔥 1. 결과 저장
    cv.imwrite("panorama_result.jpg", result)
    print("저장 완료: panorama_result.jpg")

    # 🔥 2. 화면에 맞게 축소해서 보기
    max_width = 1200
    h, w = result.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        preview = cv.resize(result, new_size)
    else:
        preview = result

    cv.imshow("Panorama (Preview)", preview)
    cv.waitKey(0)
    cv.destroyAllWindows()