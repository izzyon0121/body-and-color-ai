# main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import os
import urllib.request
from sklearn.cluster import KMeans
import mediapipe as mp

app = FastAPI()

# ==========================================
# 1. 모델 글로벌 로드 (서버 시작 시 1회)
# ==========================================
pose_model = YOLO('yolov8n-pose.pt')
seg_model = YOLO('yolov8n-seg.pt')

MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("미디어파이프 모델 다운로드 중...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)

# ==========================================
# 2. 공통 설정 및 헬퍼 함수
# ==========================================
COLOR_DETAILS = {
    "Spring Warm": {
        "name_kr": "봄 웜톤",
        "features": "고명도 / 고채도 (밝고 맑은 피부톤)",
        "best_colors": "아이보리, 피치, 코랄, 라이트 옐로우, 웜 파스텔",
        "worst_colors": "블랙, 칙칙한 다크 브라운, 푸른빛 도는 딥 그레이",
        "styling_tip": "명도가 높고 따뜻한 색상의 니트나 가디건이 찰떡!"
    },
    "Summer Cool": {
        "name_kr": "여름 쿨톤",
        "features": "고명도 / 저채도 (밝고 회끼 섞인 부드러운 피부톤)",
        "best_colors": "퓨어 화이트, 파스텔 핑크, 소라색, 라벤더, 네이비",
        "worst_colors": "오렌지, 카키, 누런 웜 브라운",
        "styling_tip": "대비가 강하지 않은 톤온톤 코디, 회끼 도는 파스텔톤 셔츠 강추!"
    },
    "Autumn Warm": {
        "name_kr": "가을 웜톤",
        "features": "저명도 / 저채도 (차분하고 부드러운 피부톤)",
        "best_colors": "베이지, 카키, 브릭 레드, 딥 브라운, 머스타드",
        "worst_colors": "핫핑크, 쨍한 블루, 형광 톤",
        "styling_tip": "가을 무드의 차분하고 딥한 얼스톤 트렌치코트나 자켓!"
    },
    "Winter Cool": {
        "name_kr": "겨울 쿨톤",
        "features": "저명도 / 고채도 (대비감이 강하거나 창백한 피부톤)",
        "best_colors": "블랙, 화이트, 버건디, 딥 블루, 비비드 핑크",
        "worst_colors": "베이지, 누런 카키, 오렌지",
        "styling_tip": "블랙&화이트처럼 대비가 뚜렷한 코디나 비비드한 원색 포인트!"
    }
}

IDX_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]


def apply_white_balance(img_rgb):
    result = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


def get_skin_color(image_rgb, mask):
    pixels = image_rgb[mask == 255]
    if len(pixels) == 0: return None
    lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    L_values = lab_pixels[:, 0]
    valid_mask = (L_values > 50) & (L_values < 200)
    filtered_pixels = pixels[valid_mask]
    if len(filtered_pixels) < 50: filtered_pixels = pixels
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42).fit(filtered_pixels)
    return kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]


def get_data(img):
    r_seg = seg_model.predict(img, conf=0.3, verbose=False)[0]
    r_pose = pose_model.predict(img, conf=0.3, verbose=False)[0]
    if r_seg.masks is None or r_pose.keypoints is None: return None, None, None
    mask = r_seg.masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
    kp = r_pose.keypoints.xy[0].cpu().numpy()
    bbox = r_seg.boxes.xyxy[0].cpu().numpy()
    return binary_mask, kp, bbox


def get_naked_width(b_mask, y, center_x, max_ratio, torso):
    if y < 0 or y >= b_mask.shape[0]: return 1
    idx = np.where(b_mask[y, :] == 255)[0]
    if len(idx) < 2: return 1
    max_dist = torso * max_ratio
    real_left = max(idx[0], center_x - max_dist)
    real_right = min(idx[-1], center_x + max_dist)
    return max(real_right - real_left, 1)


def get_w(b_mask, y):
    if y < 0 or y >= b_mask.shape[0]: return 1
    idx = np.where(b_mask[y, :] == 255)[0]
    return (idx[-1] - idx[0]) if len(idx) > 1 else 1


def resize_image(img, target_height=640):
    h, w = img.shape[:2]
    if h > target_height:
        ratio = target_height / float(h)
        new_w = int(w * ratio)
        return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return img


# ==========================================
# API 1: 체형 분석 시스템 (/analyze/body)
# ==========================================
@app.post("/analyze/body")
async def analyze_body(
        gender: str = Form(...),
        front_img: UploadFile = Form(...),
        side_img: UploadFile = Form(...)
):
    gender = gender.strip().upper()
    f_cv = cv2.imdecode(np.frombuffer(await front_img.read(), np.uint8), cv2.IMREAD_COLOR)
    s_cv = cv2.imdecode(np.frombuffer(await side_img.read(), np.uint8), cv2.IMREAD_COLOR)

    f_cv = resize_image(f_cv)
    s_cv = resize_image(s_cv)

    f_mask, f_kp, f_bbox = get_data(f_cv)
    s_mask, s_kp, s_bbox = get_data(s_cv)

    if f_mask is None or s_mask is None:
        return JSONResponse(status_code=400, content={"error": "체형 분석 실패. 사람이 제대로 나오지 않았습니다."})

    f_sy = int((f_kp[5][1] + f_kp[6][1]) / 2)
    f_hy = int((f_kp[11][1] + f_kp[12][1]) / 2)
    f_torso = max(f_hy - f_sy, 1)
    f_cx = int((f_kp[5][0] + f_kp[6][0] + f_kp[11][0] + f_kp[12][0]) / 4)

    f_s_w = get_naked_width(f_mask, f_sy + int(f_torso * 0.05), f_cx, 0.35, f_torso)
    f_w_w = get_naked_width(f_mask, f_sy + int(f_torso * 0.65), f_cx, 0.35, f_torso)
    f_h_w = get_naked_width(f_mask, f_hy + int(f_torso * 0.05), f_cx, 0.35, f_torso)

    s_to_h = f_s_w / f_h_w
    w_to_h = f_w_w / f_h_w
    w_to_s = f_w_w / f_s_w
    f_aspect = (f_bbox[2] - f_bbox[0]) / (f_bbox[3] - f_bbox[1])

    s_sy = int(max(s_kp[5][1], s_kp[6][1]))
    s_hy = int(max(s_kp[11][1], s_kp[12][1]))
    s_torso = max(s_hy - s_sy, 1)
    s_chest_d = get_w(s_mask, s_sy + int(s_torso * 0.25))
    s_belly_d = get_w(s_mask, s_sy + int(s_torso * 0.65))
    side_fat_ratio = s_belly_d / s_chest_d

    body_type = ""
    if gender == 'M':
        if f_aspect < 0.38:
            body_type = "The Lean Column"
        elif (side_fat_ratio > 1.02 or f_aspect > 0.48) and w_to_h > 0.90:
            body_type = "The Apple"
        elif side_fat_ratio < 0.98 and s_to_h > 1.18 and w_to_s < 0.80:
            body_type = "The Inverted Triangle"
        elif s_to_h < 0.98 and w_to_h < 0.95:
            body_type = "The Pear"
        else:
            if f_w_w / ((f_s_w + f_h_w) / 2) < 0.85:
                body_type = "The Hour Glass"
            else:
                body_type = "The Rectangle"
    elif gender == 'F':
        if (side_fat_ratio > 1.02 or f_aspect > 0.45) and w_to_h > 0.92:
            body_type = "The Apple"
        elif f_w_w / ((f_s_w + f_h_w) / 2) < 0.78 and abs(f_s_w - f_h_w) < (f_h_w * 0.1):
            body_type = "The Hour Glass"
        elif s_to_h < 0.92:
            body_type = "The Pear"
        elif s_to_h > 1.08:
            body_type = "The Inverted Triangle"
        else:
            if f_aspect < 0.38:
                body_type = "The Lean Column"
            else:
                body_type = "The Rectangle"

    return {
        "gender": gender,
        "body_type": body_type,
        "metrics": {"f_aspect": float(f_aspect), "side_fat_ratio": float(side_fat_ratio)}
    }


# ==========================================
# API 2: 퍼스널 컬러 분석 시스템 (/analyze/color)
# ==========================================
@app.post("/analyze/color")
async def analyze_color(front_img: UploadFile = Form(...)):
    front_bytes = await front_img.read()
    f_cv_orig = cv2.imdecode(np.frombuffer(front_bytes, np.uint8), cv2.IMREAD_COLOR)

    h, w, _ = f_cv_orig.shape
    img_rgb = cv2.cvtColor(f_cv_orig, cv2.COLOR_BGR2RGB)
    wb_img_rgb = apply_white_balance(img_rgb)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=wb_img_rgb)
    result = landmarker.detect(mp_image)

    final_tone = "Unknown"

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        mask_skin = np.zeros((h, w), dtype=np.uint8)
        pts_skin = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in IDX_CHEEK], np.int32)
        cv2.fillPoly(mask_skin, [pts_skin], 255)

        current_skin = get_skin_color(wb_img_rgb, mask_skin)

        if current_skin is not None:
            lab = cv2.cvtColor(np.uint8([[current_skin]]), cv2.COLOR_RGB2Lab)[0][0]
            hsv = cv2.cvtColor(np.uint8([[current_skin]]), cv2.COLOR_RGB2HSV)[0][0]
            L, a, b = lab
            _, S, _ = hsv

            TH_B, TH_L, TH_S = 128, 150, 45

            if b > TH_B:
                if L > TH_L and S > TH_S:
                    final_tone = "Spring Warm"
                else:
                    final_tone = "Autumn Warm"
            else:
                if L > TH_L and S <= TH_S:
                    final_tone = "Summer Cool"
                else:
                    final_tone = "Winter Cool"

    if final_tone == "Unknown":
        return JSONResponse(status_code=400, content={"error": "퍼스널 컬러 분석 실패. 얼굴을 명확히 인식할 수 없습니다."})

    return {
        "personal_color": final_tone,
        "color_details": COLOR_DETAILS[final_tone]
    }
