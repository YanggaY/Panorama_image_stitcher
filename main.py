import numpy as np
import cv2 as cv

#이미지 로드
img1 = cv.imread('data/1.jpeg')
img2 = cv.imread('data/2.jpeg')
img3 = cv.imread('data/3.jpeg')

assert (img1 is not None) and (img2 is not None) and (img3 is not None), 'Cannot read images'

#BRISK 검출기, brute-force matcher(Hamming 거리 사용)
brisk = cv.BRISK_create()
fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')

#stitch_two : 두 이미지 이어붙이는 함수
#img_b를 img_a 좌표계로 변환해서 합친 결과를 반환
def stitch_two(img_a, img_b):
    #특징점 검출
    keypoints1, descriptors1 = brisk.detectAndCompute(img_a, None)
    keypoints2, descriptors2 = brisk.detectAndCompute(img_b, None)

    #매칭
    match = fmatcher.match(descriptors1, descriptors2)

    #매칭된 점의 좌표 추출
    pts_a = np.float32([keypoints1[m.queryIdx].pt for m in match])
    pts_b = np.float32([keypoints2[m.trainIdx].pt for m in match])

    #호모그래피 추정(RANSAC)
    H, inlier_mask = cv.findHomography(pts_b, pts_a, cv.RANSAC, 3.0)

    #캔버스크기 계산
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    corners_b_warped = cv.perspectiveTransform(corners_b, H)

    all_corners = np.concatenate([corners_a, corners_b_warped], axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    #음수 좌표 보정
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float64)
    canvas_w, canvas_h = x_max - x_min, y_max - y_min

    #img_a, img_b를 각각 캔버스에 배치
    warped_a = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    warped_a[-y_min:-y_min + h_a, -x_min:-x_min + w_a] = img_a

    warped_b = cv.warpPerspective(img_b, T @ H, (canvas_w, canvas_h))

    #유효영역 마스크 만들기(img_a가 있는곳은 255 나머지는 0으로 구분)
    mask_a = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_a[-y_min:-y_min + h_a, -x_min:-x_min + w_a] = 255

    mask_b_src = np.ones((h_b, w_b), dtype=np.uint8) * 255
    mask_b = cv.warpPerspective(mask_b_src, T @ H, (canvas_w, canvas_h))

    #Distance transform으로 가중치 만들기
    weight_a = cv.distanceTransform(mask_a, cv.DIST_L2, 5).astype(np.float32)
    weight_b = cv.distanceTransform(mask_b, cv.DIST_L2, 5).astype(np.float32)

    #가중평균으로 blending(픽셀값*가중치)
    acc = warped_a.astype(np.float32) * weight_a[..., None] + warped_b.astype(np.float32) * weight_b[..., None]
    weight_sum = weight_a + weight_b
    weight_sum[weight_sum == 0] = 1.0  #0으로 나누는것을 방지

    result = (acc / weight_sum[..., None]).clip(0, 255).astype(np.uint8)

    # 검은 여백 제거
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.concatenate(contours, axis=0)
        x, y, w, h = cv.boundingRect(all_pts)
        result = result[y:y+h, x:x+w]

    return result


step1 = stitch_two(img1, img2)

step2 = stitch_two(step1, img3)

cv.imwrite('Result.jpg', step2)
print("Result has been saved as Result.jpg")
cv.imshow('Step1 Result', step1)
cv.imshow('Step2 Result', step2)
cv.waitKey(0)
cv.destroyAllWindows()
