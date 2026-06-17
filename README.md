# AI Fashion Scanner: Body & Color Analyzer

> 중고 의류 거래 플랫폼 **CLO-VER**의 핵심 AI 비전 모듈 (개인 포트폴리오 저장소)

## 프로젝트 개요 (Overview)
기존 중고 의류 거래의 가장 큰 페인포인트(Pain Point)인 '사이즈 미스'와 '스타일 부조화'를 해결하기 위해 기획된 **AI 기반 체형 및 퍼스널 컬러 분석 솔루션**입니다. 
스마트폰 카메라로 촬영한 사진 데이터를 기반으로 사용자의 체형을 스캔하고 피부톤을 분석하여, 실패 없는 최적의 중고 의류 매칭 경험을 제공합니다.

본 레포지토리는 졸업 프로젝트인 종합 중고 의류 거래 플랫폼 'CLO-VER'의 전체 시스템 중, 본인이 직접 기획하고 전담하여 개발한 **AI 비전 처리 및 데이터 분석 API 모듈**만을 분리한 저장소입니다.

## 핵심 기능 (Key Features)

### 1. AI 바디 스캐너 (Body Shape Analyzer)
- **동작 원리:** 이미지 비전 기술을 활용하여 사용자의 전신 사진에서 주요 관절 및 신체 윤곽선(Landmark)을 추출합니다.
- **핵심 기술:** [사용된 기술, 예: MediaPipe / OpenCV / YOLOv8 등]을 활용한 신체 비율 및 수치 추론.
- **기대 효과:** 단순히 'S/M/L' 사이즈 표기를 넘어, 어깨너비, 기장 등 세밀한 체형 데이터를 기반으로 사용자에게 딱 맞는 의류 핏을 계산합니다.

### 2. 퍼스널 컬러 추출기 (Personal Color Extractor)
- **동작 원리:** 사용자 이미지에서 얼굴 영역을 정확히 인식한 후, 픽셀 색상 데이터를 분석하여 웜/쿨톤 및 4계절 퍼스널 컬러를 도출합니다.
- **핵심 기술:** [예: Dlib/OpenCV를 통한 얼굴 인식 및 K-Means 클러스터링 알고리즘]을 적용한 피부/머리/눈동자 색상 추출.
- **기대 효과:** 사용자 본인의 퍼스널 컬러에 최적화된 옷 색상을 추천하여 중고 의류 구매 후 만족도를 극대화합니다.

## 기술 스택 (Tech Stack)
- **AI / Computer Vision:** `Python`, `OpenCV`, `[PyTorch / TensorFlow / MediaPipe 등]`
- **Backend / API:** `[FastAPI 또는 Flask / Spring Boot 등 추론 서버 구성에 쓴 기술]`
- **Architecture:** AI 모델의 추론 결과를 메인 서비스(CLO-VER) 프론트엔드로 빠르고 안정적으로 전달하기 위한 RESTful API 설계

## 나의 역할 및 성과 (My Role & Impact)
* **PM 및 서비스 기획:** 단순한 기술 구현을 넘어, '중고 의류 거래 실패율 감소'라는 비즈니스 목표를 달성하기 위한 AI 파이프라인 설계
* **AI 모델 최적화:** 실서비스 환경을 고려하여 처리 속도와 정확도의 밸런스를 맞춘 가벼운 추론 로직 구현
* **API 연동:** 도출된 체형 및 컬러 분석 데이터를 메인 백엔드(Spring Boot) 및 데이터베이스(MySQL)와 유기적으로 연동

## 시연 및 실행 화면 (Demo)
-체형분석
<img width="400" height="213" alt="Image" src="https://github.com/user-attachments/assets/5bd240e4-b67d-4936-94ca-03fb9f3d58c8" />

-퍼스널 컬러
https://github.com/izzyon0121/body-and-color-ai/issues/2#issue-4685070058


