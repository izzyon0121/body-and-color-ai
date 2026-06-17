import os
import urllib.request
import cv2
import numpy as np
from sklearn.cluster import KMeans
import mediapipe as mp
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import textwrap  # 동적 텍스트 렌더링 시 레이아웃 이탈 방지 및 자동 개행 처리를 위한 모듈

# ==========================================
# 퍼스널 컬러 4계절 분기별 메타데이터 및 스타일링 추천 데이터베이스
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

# ==========================================
# 1. MediaPipe Face Landmarker 추론 모델 가중치 파일 로드 및 초기화
# ==========================================
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("미디어파이프 모델 다운로드 중...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", MODEL_PATH)

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
# 2. 핵심 알고리즘: 이미지 전처리 및 K-Means 기반 주요 색상(Dominant Color) 군집화
# ==========================================
def apply_white_balance(img_rgb):
    # Gray World Assumption 기반의 Color Constancy(색채 항상성) 알고리즘 적용
    # 주변 광원(조명)의 영향을 상쇄하여 피사체 본연의 반사율(Reflectance)을 추정함
    result = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

def get_skin_color(image_rgb, mask):
    # 피부 영역(볼, 턱선) 내부 픽셀 추출 및 전처리 프로세스
    pixels = image_rgb[mask == 255]
    if len(pixels) == 0: return None
    lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    L_values = lab_pixels[:, 0]
    # L(명도) 채널 기준 극단값 필터링: 음영(Shadow, < 50) 및 난반사(Highlight, > 200) 노이즈 제거
    valid_mask = (L_values > 50) & (L_values < 200) 
    filtered_pixels = pixels[valid_mask]
    # 유효 픽셀 수가 임계치 이하일 경우 Fallback(원본 픽셀) 사용
    if len(filtered_pixels) < 50: filtered_pixels = pixels
    # K-Means 클러스터링을 통한 최적의 대표 스킨 컬러 도출
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42).fit(filtered_pixels)
    return kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

def get_feature_color(image_rgb, mask):
    # 특정 이목구비(입술, 눈동자) 내부 픽셀의 대표 색상 군집화
    pixels = image_rgb[mask == 255]
    if len(pixels) == 0: return None
    kmeans = KMeans(n_clusters=2, n_init=5, random_state=42).fit(pixels)
    return kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

def draw_korean_text(img, text, position, font_size, text_color):
    # OpenCV의 멀티바이트 문자(한글) 렌더링 한계를 극복하기 위한 Pillow(PIL) 변환 헬퍼 함수
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        # 시스템 기본 내장 폰트(Windows: 맑은 고딕) 로드 시도
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        # 폰트 부재 시 크로스플랫폼 호환성을 위한 기본 폰트 Fallback
        font = ImageFont.load_default() 
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def save_result_to_db(user_id, final_tone, color_data):
    # 분석된 퍼스널 컬러 스펙과 추출된 Feature 기반의 백엔드(REST API) 전송용 JSON 페이로드 구성
    payload = {
        "user_id": user_id,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "personal_color": final_tone,
        "details": COLOR_DETAILS[final_tone],
        "extracted_rgb": {
            "skin": [int(c) for c in color_data['skin']] if color_data['skin'] is not None else [],
            "lips": [int(c) for c in color_data['lips']] if color_data['lips'] is not None else [],
            "eyes": [int(c) for c in color_data['eyes']] if color_data['eyes'] is not None else []
        }
    }
    print("\n" + "DB 업로드 페이로드 생성".center(50, "="))
    print(json.dumps(payload, indent=4, ensure_ascii=False))
    print("==================================================")

# ==========================================
# 3. 실시간 웹캠 스트리밍 기반 비전 AI 파이프라인 (안면 인식 및 진단)
# ==========================================
cap = cv2.VideoCapture(0)
print("CLO-VER 실시간 퍼스널 컬러 풀스캐너 켜짐!")
print("카메라를 정면으로 바라보세요. 3초 후 자동으로 확정됩니다!\n")

# MediaPipe Face Mesh 기준 부위별 랜드마크(Landmark) 인덱스 정의
IDX_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
IDX_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
IDX_L_EYE = [33, 160, 158, 133, 153, 144]
IDX_R_EYE = [362, 385, 387, 263, 373, 380]

# UX 개선: 안면 정면 인식률 향상을 위한 상태 기반(State-based) 3초 카운트다운 타이머 변수
start_time = None
countdown_duration = 3  

# 스레드 종료 후 데이터 보존을 위한 전역 스코프 캐싱(Caching) 변수
final_captured_tone = None
final_captured_colors = {}
captured_frame = None  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 사용자 직관성(Mirror Effect) 확보를 위한 비디오 프레임 수평 반전
    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    wb_img_rgb = apply_white_balance(img_rgb)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=wb_img_rgb)
    result = landmarker.detect(mp_image)
    
    display_text = "Scanning..."
    color_box = (255, 255, 255)
    
    current_skin, current_lips, current_eyes = None, None, None

    if result.face_landmarks:
        # 안면 인식 감지(Trigger) 시 타이머 프로비저닝 시작
        if start_time is None:
            start_time = time.time()

        elapsed_time = time.time() - start_time
        remaining_time = max(0, int(countdown_duration - elapsed_time + 1))

        landmarks = result.face_landmarks[0]
        
        # 3.1 피부 영역(Cheek/Jawline) ROI 마스킹 및 특징 추출
        mask_skin = np.zeros((h, w), dtype=np.uint8)
        pts_skin = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in IDX_CHEEK], np.int32)
        cv2.fillPoly(mask_skin, [pts_skin], 255)
        cv2.polylines(frame, [pts_skin], True, (0, 255, 0), 2) 
        current_skin = get_skin_color(wb_img_rgb, mask_skin)

        # 3.2 입술 영역(Lips) ROI 마스킹 및 특징 추출
        mask_lips = np.zeros((h, w), dtype=np.uint8)
        pts_lips = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in IDX_LIPS], np.int32)
        cv2.fillPoly(mask_lips, [pts_lips], 255)
        cv2.polylines(frame, [pts_lips], True, (0, 0, 255), 2) 
        current_lips = get_feature_color(wb_img_rgb, mask_lips)

        # 3.3 안구 영역(Eyes) ROI 병합 마스킹 및 특징 추출
        mask_eyes = np.zeros((h, w), dtype=np.uint8)
        pts_l_eye = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in IDX_L_EYE], np.int32)
        pts_r_eye = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in IDX_R_EYE], np.int32)
        cv2.fillPoly(mask_eyes, [pts_l_eye], 255)
        cv2.fillPoly(mask_eyes, [pts_r_eye], 255)
        cv2.polylines(frame, [pts_l_eye], True, (255, 0, 0), 2) 
        cv2.polylines(frame, [pts_r_eye], True, (255, 0, 0), 2)
        current_eyes = get_feature_color(wb_img_rgb, mask_eyes)

        # 3.4 휴리스틱(Heuristic) 기반 4계절 톤 판별 매트릭스 실행
        if current_skin is not None:
            # 다차원 색공간(CIELAB, HSV) 교차 검증을 통한 파라미터 분리
            lab = cv2.cvtColor(np.uint8([[current_skin]]), cv2.COLOR_RGB2Lab)[0][0]
            hsv = cv2.cvtColor(np.uint8([[current_skin]]), cv2.COLOR_RGB2HSV)[0][0]
            L, a, b = lab 
            _, S, _ = hsv

            # 모델 캘리브레이션을 위한 하이퍼파라미터 (b:황색도, L:명도, S:채도/청탁)
            TH_B, TH_L, TH_S = 128, 150, 45   

            # 결정 트리(Decision Tree) 구조의 계절별 분기 처리
            if b > TH_B:
                if L > TH_L and S > TH_S: display_text, color_box = "Spring Warm", (0, 165, 255)
                else: display_text, color_box = "Autumn Warm", (34, 139, 34)
            else:
                if L > TH_L and S <= TH_S: display_text, color_box = "Summer Cool", (255, 192, 203)
                else: display_text, color_box = "Winter Cool", (255, 0, 0)

            # 분석된 주요 컬러 팔레트(Swatches)의 실시간 GUI 렌더링
            swatches = [("Skin", current_skin, 50), ("Lips", current_lips, 100), ("Eyes", current_eyes, 150)]
            for name, rgb, y_pos in swatches:
                if rgb is not None:
                    # RGB to BGR 배열 구조 변환 후 OpenCV 사각형 작도
                    bgr_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                    cv2.rectangle(frame, (10, y_pos), (50, y_pos+30), bgr_color, -1)
                    cv2.putText(frame, name, (60, y_pos+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 사용자 정면 응시 유도를 위한 시각적(Visual) 카운트다운 오버레이
        if remaining_time > 0:
            cv2.putText(frame, str(remaining_time), (w // 2 - 30, h // 2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, "LOOK AT THE CAMERA!", (w // 2 - 180, h // 2 + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 카운트다운 만료 시(Timeout) 현재의 분석 결과값을 스냅샷으로 확정 및 메모리 캐싱
        if elapsed_time >= countdown_duration:
            if display_text != "Scanning...":
                final_captured_tone = display_text
                final_captured_colors = {"skin": current_skin, "lips": current_lips, "eyes": current_eyes}
                captured_frame = frame.copy() 
                break
    else:
        # 안면 인식(Landmark Detection) 실패 시 예외 처리: 카운트다운 컨텍스트 초기화
        start_time = None

    # 상태별(State) 상단 헤더 텍스트 실시간 렌더링
    cv2.putText(frame, f"Tone: {display_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color_box, 3, cv2.LINE_AA)
    cv2.imshow('CLO-VER AI Full Scanner', frame)

    # I/O 인터럽트 감지: Q 입력 시 메인 루프 강제 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("중도 종료됨.")
        break

# 시스템 자원(카메라 디바이스) 메모리 누수 방지 및 반환
cap.release()
cv2.destroyAllWindows()

# ==========================================
# 4. 진단 완료 후 최종 결과 리포트 대시보드 렌더링 및 UI 표출
# ==========================================
if final_captured_tone:
    # 캡처된 데이터를 기반으로 백엔드 저장 API 호출 트리거
    save_result_to_db(user_id="user_clover_001", final_tone=final_captured_tone, color_data=final_captured_colors)
    
    # 캐싱된 진단 메타데이터 로드
    info = COLOR_DETAILS[final_captured_tone]
    
    # UI 레이아웃 설계: 텍스트 가독성 확보를 위한 600px 패널 할당 및 최소 Y축 해상도(780px) 동적 보장
    report_panel_width = 600
    h, w, _ = captured_frame.shape
    board_h = max(h, 780) 
    
    # 통합 뷰포트용 캔버스 메모리 할당 (기본값: Black)
    result_board = np.zeros((board_h, w + report_panel_width, 3), dtype=np.uint8)
    
    # 뷰포트 좌측(Left Area): 공간 여백 처리 및 원본 스냅샷 이미지 매핑
    result_board[:, :w] = (30, 30, 30)
    result_board[0:h, 0:w] = captured_frame
    
    # 뷰포트 우측(Right Area): 다크 모드(Dark Mode) 컨셉의 텍스트 패널 배경 렌더링
    result_board[:, w:] = (40, 40, 40)
    
    # 계층형(Hierarchical) 텍스트 렌더링: 메인 타이틀 및 구분선 표출
    result_board = draw_korean_text(result_board, f"[AI 퍼스널 컬러 진단 리포트]", (w + 20, 30), 26, (255, 255, 255))
    result_board = draw_korean_text(result_board, f"----------------------------------------------------------", (w + 20, 75), 16, (120, 120, 120))
    
    # 계층형 텍스트 렌더링: 코어 진단 결과 표출
    result_board = draw_korean_text(result_board, f"최종 확정 톤", (w + 20, 105), 20, (0, 255, 255))
    result_board = draw_korean_text(result_board, f"- {info['name_kr']} ({final_captured_tone})", (w + 20, 135), 24, (255, 255, 255))
    
    result_board = draw_korean_text(result_board, f"피부 톤 특징", (w + 20, 195), 20, (150, 255, 150))
    result_board = draw_korean_text(result_board, f"- {info['features']}", (w + 20, 225), 16, (230, 230, 230))
    
    # 모듈화된 텍스트 렌더링 적용: 긴 문장 데이터의 뷰포트 이탈 방지를 위한 자동 개행 알고리즘(textwrap) 수행
    result_board = draw_korean_text(result_board, f"CLO-VER 추천 베스트 컬러", (w + 20, 285), 20, (150, 150, 255))
    best_colors_wrapped = textwrap.wrap(info['best_colors'], width=35)
    for idx, line in enumerate(best_colors_wrapped):
        result_board = draw_korean_text(result_board, f"- {line}", (w + 20, 315 + (idx * 30)), 16, (255, 255, 255))
        
    # UI 컴포넌트의 동적 배치: 선행된 텍스트 블록의 높이(Height)를 계산하여 후속 요소의 Y축 좌표 포지셔닝
    next_y = 315 + (len(best_colors_wrapped) * 30) + 30
    
    result_board = draw_korean_text(result_board, f"피해야 할 워스트 컬러", (w + 20, next_y), 20, (255, 150, 150))
    worst_colors_wrapped = textwrap.wrap(info['worst_colors'], width=35)
    for idx, line in enumerate(worst_colors_wrapped):
        result_board = draw_korean_text(result_board, f"- {line}", (w + 20, next_y + 30 + (idx * 30)), 16, (240, 200, 200))
        
    next_y = next_y + 30 + (len(worst_colors_wrapped) * 30) + 30
    
    result_board = draw_korean_text(result_board, f"중고 의류 코디/스타일링 팁", (w + 20, next_y), 20, (255, 255, 150))
    styling_tip_wrapped = textwrap.wrap(info['styling_tip'], width=35)
    for idx, line in enumerate(styling_tip_wrapped):
        result_board = draw_korean_text(result_board, f"- {line}", (w + 20, next_y + 30 + (idx * 30)), 16, (240, 240, 240))

    # Event Loop: 최종 렌더링된 대시보드를 홀드 상태로 유지하며 사용자 입력(Exit) 대기
    while True:
        cv2.imshow('CLO-VER AI Diagnosis Result (Press Q to Exit)', result_board)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()