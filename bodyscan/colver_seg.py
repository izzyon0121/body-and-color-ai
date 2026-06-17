import cv2
import numpy as np
import time
from ultralytics import YOLO

print("체형분석 시작")

# ==========================================
# 성별 입력
# ==========================================
print("\n유저 정보 입력")
while True:
    gender = input("성별을 입력하시오 (남자: M, 여자: F 입력): ").strip().upper()
    if gender in ['M', 'F']:
        break
    print("다시 입력하시오")

print(f"성별 [{gender}] 확인 완료.\n")

# 1. 모델 로딩
pose_model = YOLO('yolov8n-pose.pt')
seg_model = YOLO('yolov8n-seg.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라가 없습니다")
    exit()

# 상태 관리 변수
capture_stage = 0  # 0: 정면, 1: 측면, 2: 분석
pose_start_time = None
capture_delay = 3.0

front_img = None
side_img = None

print("[STEP 1] 정면을 보고 양손을 3초간 들어주세요.")

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    im_h, im_w = frame.shape[:2]

    if capture_stage == 2:
        break

    results = pose_model.predict(frame, conf = 0.5, verbose = False)
    annotated_frame = results[0].plot()

    guide_text = "STEP 1: FRONT VIEW (Hands UP!)" if capture_stage == 0 else "STEP 2: SIDE VIEW (Turn & Hands UP!)"
    box_color = (255, 200, 0) if capture_stage == 0 else (0, 200, 255)

    cv2.rectangle(annotated_frame, (int(im_w * 0.2), int(im_h * 0.1)), (int(im_w * 0.8), int(im_h * 0.9)), box_color, 2)
    cv2.putText(annotated_frame, guide_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0 and len(results[0].keypoints.xy[0]) > 10:
        kp_live = results[0].keypoints.xy[0].cpu().numpy()

        l_s_y, r_s_y = kp_live[5][1], kp_live[6][1]
        l_w_y, r_w_y = kp_live[9][1], kp_live[10][1]

        hands_up = False
        if (l_w_y > 0 and l_s_y > 0 and l_w_y < l_s_y) or (r_w_y > 0 and r_s_y > 0 and r_w_y < r_s_y):
            hands_up = True

        if hands_up:
            if pose_start_time is None: pose_start_time = time.time()
            elapsed = time.time() - pose_start_time
            remain = max(0, capture_delay - elapsed)

            cv2.putText(annotated_frame, f"HOLD: {remain:.1f}s", (int(im_w / 2) - 100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 4)
            cv2.rectangle(annotated_frame, (int(im_w * 0.2), int(im_h * 0.1)), (int(im_w * 0.8), int(im_h * 0.9)),
                          (0, 255, 0), 4)

            if remain == 0:
                if capture_stage == 0:
                    print("\n정면 캡처 완료! 옆으로 돌아주세요.")
                    front_img = frame.copy()
                    capture_stage = 1
                    pose_start_time = None
                    cv2.putText(annotated_frame, "FRONT CAPTURED! TURN SIDE", (50, int(im_h / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                    cv2.imshow("Clover AI Live", annotated_frame)
                    cv2.waitKey(1500)

                elif capture_stage == 1:
                    print("\n측면 캡처 완료! 성별 맞춤형 분석을 시작합니다...")
                    side_img = frame.copy()
                    capture_stage = 2
                    cv2.putText(annotated_frame, "ALL CAPTURED! ANALYZING...", (50, int(im_h / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                    cv2.imshow("Clover AI Live", annotated_frame)
                    cv2.waitKey(500)
        else:
            pose_start_time = None
    else:
        pose_start_time = None

    cv2.imshow("Clover AI Live", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# [STEP 3] 안티 클로딩 & 남녀 분기 정밀 분석 로직
# ==========================================
if capture_stage == 2 and front_img is not None and side_img is not None:
    print("\n[AI] 옷 두께 제거 및 데이터 추출 중...")


    def get_data(img):
        r_seg = seg_model.predict(img, conf = 0.3, verbose = False)[0]
        r_pose = pose_model.predict(img, conf = 0.3, verbose = False)[0]
        if r_seg.masks is None or r_pose.keypoints is None: return None, None, None

        mask = r_seg.masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        # 침식 연산 (옷 두께 깎기)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.erode(binary_mask, kernel, iterations = 2)

        kp = r_pose.keypoints.xy[0].cpu().numpy()
        bbox = r_seg.boxes.xyxy[0].cpu().numpy()
        return binary_mask, kp, bbox


    f_mask, f_kp, f_bbox = get_data(front_img)
    s_mask, s_kp, s_bbox = get_data(side_img)

    if f_mask is not None and s_mask is not None:
        fh, fw = front_img.shape[:2]

        f_sy = int((f_kp[5][1] + f_kp[6][1]) / 2)
        f_hy = int((f_kp[11][1] + f_kp[12][1]) / 2)
        f_torso = max(f_hy - f_sy, 1)
        f_cx = int((f_kp[5][0] + f_kp[6][0] + f_kp[11][0] + f_kp[12][0]) / 4)


        # 뼈대 한계선 기반 너비 측정 (옷 펄럭임 고려해서 수정)
        def get_naked_width(b_mask, y, center_x, max_ratio, torso):
            if y < 0 or y >= b_mask.shape[0]: return 1
            idx = np.where(b_mask[y, :] == 255)[0]
            if len(idx) < 2: return 1
            max_dist = torso * max_ratio
            real_left = max(idx[0], center_x - max_dist)
            real_right = min(idx[-1], center_x + max_dist)
            return max(real_right - real_left, 1)


        f_s_w = get_naked_width(f_mask, f_sy + int(f_torso * 0.05), f_cx, 0.35, f_torso)
        f_c_w = get_naked_width(f_mask, f_sy + int(f_torso * 0.25), f_cx, 0.35, f_torso)
        f_w_w = get_naked_width(f_mask, f_sy + int(f_torso * 0.65), f_cx, 0.35, f_torso)
        f_h_w = get_naked_width(f_mask, f_hy + int(f_torso * 0.05), f_cx, 0.35, f_torso)

        s_to_h = f_s_w / f_h_w
        w_to_h = f_w_w / f_h_w
        w_to_s = f_w_w / f_s_w
        f_aspect = (f_bbox[2] - f_bbox[0]) / (f_bbox[3] - f_bbox[1])

        sh, sw = side_img.shape[:2]
        s_sy = int(max(s_kp[5][1], s_kp[6][1]))
        s_hy = int(max(s_kp[11][1], s_kp[12][1]))
        s_torso = max(s_hy - s_sy, 1)


        def get_w(b_mask, y):
            if y < 0 or y >= b_mask.shape[0]: return 1
            idx = np.where(b_mask[y, :] == 255)[0]
            return (idx[-1] - idx[0]) if len(idx) > 1 else 1


        s_chest_d = get_w(s_mask, s_sy + int(s_torso * 0.25))
        s_belly_d = get_w(s_mask, s_sy + int(s_torso * 0.65))
        side_fat_ratio = s_belly_d / s_chest_d

        print(f"\n덩치:{f_aspect:.2f} | 측면비율:{side_fat_ratio:.2f} | 어깨/골반:{s_to_h:.2f} | 허리/어깨:{w_to_s:.2f}")

        # ==========================================
        # 성별에 따른 분류
        # ==========================================
        if gender == 'M':
            print("[남자 기준] 체형 분석을 시작합니다.")
            if f_aspect < 0.38:
                body_type = "The Lean Column"
                color = (255, 255, 0)
            elif (side_fat_ratio > 1.02 or f_aspect > 0.48) and w_to_h > 0.90:
                body_type = "The Apple"
                color = (0, 0, 255)
            elif side_fat_ratio < 0.98 and s_to_h > 1.18 and w_to_s < 0.80:
                body_type = "The Inverted Triangle"
                color = (0, 255, 0)
            elif s_to_h < 0.98 and w_to_h < 0.95:
                body_type = "The Pear"
                color = (0, 165, 255)
            else:
                if f_w_w / ((f_s_w + f_h_w) / 2) < 0.85:
                    body_type = "The Hour Glass"
                    color = (255, 0, 255)
                else:
                    body_type = "The Rectangle"
                    color = (255, 255, 0)

        elif gender == 'F':
            print("[여자 기준] 체형 분석을 시작합니다.")
            if (side_fat_ratio > 1.02 or f_aspect > 0.45) and w_to_h > 0.92:
                body_type = "The Apple"
                color = (0, 0, 255)
            elif f_w_w / ((f_s_w + f_h_w) / 2) < 0.78 and abs(f_s_w - f_h_w) < (f_h_w * 0.1):
                body_type = "The Hour Glass"
                color = (255, 0, 255)
            elif s_to_h < 0.92:
                body_type = "The Pear"
                color = (0, 165, 255)
            elif s_to_h > 1.08:
                body_type = "The Inverted Triangle"
                color = (0, 255, 0)
            else:
                if f_aspect < 0.38:
                    body_type = "The Lean Column"
                else:
                    body_type = "The Rectangle"
                color = (255, 255, 0)

        # 결과 출력
        print(f"==>  최종 판정: {body_type}")

        cv2.putText(front_img, f"Gender: {gender} | {body_type}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(side_img, f"Side Fat Ratio: {side_fat_ratio:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        combined_img = np.hstack((cv2.resize(front_img, (500, 700)), cv2.resize(side_img, (500, 700))))
        cv2.imshow('Closeter Ultimate V9', combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("분석 실패! 사진이 흔들렸거나 사람이 화면에 제대로 나오지 않았습니다.")
