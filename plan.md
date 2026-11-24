# 드론 탐지 모델 플랜 (YOLO26n / YOLO26s)

## 1. 목표
- **지상→하늘**(지상 카메라로 하늘의 드론 탐지)과 **공중→공중**(비행 중인 드론이 다른 드론 탐지)을 모두 처리하는 단일 탐지 모델
- **YOLO26n**과 **YOLO26s**를 **같은 데이터, 같은 설정**으로 학습해서 비교
- 무료이고 라이선스가 비교적 깔끔한 공개 데이터셋만 사용

> **2026-11-20 샘플 확인 후 변경**: 공중→공중 주 테스트를 Det-Fly 장면 그룹 홀드아웃으로 바꾸고 AOT 드론 비행 7개는 보조 테스트로, AOT는 오탐·배경 소스로 역할을 바꿨다. AOD-4의 YouTube·Anti-UAV 출신 이미지와 seraphim 안의 pathikg 이미지는 제외한다. 자세한 내용과 이유는 `spec.md` 0절, `notes/sample-check.md`. 아래 표와 충돌하면 spec.md 0절이 우선한다.

## 2. 확정 사항 요약
| 항목 | 결정 |
|---|---|
| 모델 | YOLO26n, YOLO26s (NMS-free, COCO 사전학습 가중치에서 시작) |
| 학습 방식 | 2단계: ① seraphim 사전학습(640) → ② 혼합 데이터 본학습(1280) |
| 클래스 | 4개: `0 drone`, `1 bird`, `2 airplane`, `3 helicopter` (추론 때는 drone만 사용) |
| 테스트셋 | 지상→하늘: Anti-UAV300 RGB / 공중→공중: AOT 홀드아웃 시퀀스 |
| 비교 기준 | `sapoepsilon/yolov11s-drone-detector` (Anti-UAV-RGBT 기준 AP@0.5 0.741) |

## 3. 데이터셋

### 학습용
| 데이터셋 | 역할 | 규모 | 라이선스 | 링크 |
|---|---|---|---|---|
| seraphim | 1단계 사전학습, 2단계 replay | 83,483장 (train 75,134 / test 8,349), YOLO 포맷, 640×640 패딩 | CC BY 4.0 | https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset |
| DUT Anti-UAV | 지상→하늘 | 10,000장 (5,200 / 2,600 / 2,200), 수동 라벨 | 레포 Apache-2.0 (데이터 조건 미명시) | https://github.com/wangdongdut/DUT-Anti-UAV |
| Det-Fly | 공중→공중 | 13,000장 이상, 4K, DJI Mavic 1종 | 레포 MIT (데이터 조건 미명시) | https://github.com/Jake-WU/Det-Fly |
| AOT (Amazon) | 공중→공중 | 4,943 시퀀스 / 590만 장 이상 → 드론 프레임만 일부 추출 | CDLA-Permissive 1.0 | https://registry.opendata.aws/airborne-object-tracking/ |
| AOD-4 | 오탐 대상(새·비행기·헬기) | 22,516장, 4클래스, 1024×1024 | CC BY 4.0 | https://universe.roboflow.com/airborne-object-detection/airborne-object-detection-4-aod4 |

### 테스트 전용 (학습·검증에 절대 사용 금지)
| 데이터셋 | 시점 | 링크 |
|---|---|---|
| Anti-UAV300 (RGB 프레임만) | 지상→하늘 | https://github.com/ZhaoJ9014/Anti-UAV |
| AOT 홀드아웃 시퀀스 | 공중→공중 | 위 AOT 링크 |

### 제외한 데이터셋과 이유
- **pathikg/drone-detection-dataset**: 유튜브 영상 프레임이라 저작권이 애매하고, 라벨 일부가 자동 생성
- **Drone-vs-Bird**: 이용 조건 확인이 필요하고 새 라벨이 없음
- **Anti-UAV410 / 600**: 열화상(IR)만 있음
- **Kaggle drone-yolo-detection / Drone Dataset (UAV)**: seraphim에 이미 포함됐을 가능성이 높음
- **Roboflow 소규모 프로젝트들**: 품질 편차가 크고 재업로드 중복이 많음
- **Det-Fly Roboflow 사본**: 연구용·비상업 조건이 붙어 있어서 원본만 사용

## 4. 학습 플로우

### 1단계: 사전학습
- 데이터: seraphim train (val은 seraphim test)
- imgsz 640. 데이터가 640 패딩이라 해상도를 올려도 이득이 없음
- 시작점: `yolo26n.pt`, `yolo26s.pt`

### 2단계: 본학습
- 시작점: 1단계 best 가중치
- 데이터: DUT + Det-Fly + AOT 드론 프레임 + AOD-4 + seraphim 일부(망각 방지용 replay)
- imgsz 1280 (작은 드론 대응, 640 대비 연산량 약 4배)
- 시점 균형: 지상→하늘과 공중→공중 비율이 한쪽으로 크게 치우치지 않게 맞추기
- 배경 이미지: 빈 하늘, 새만 있는 프레임을 라벨 없이 소량 포함

### 선택 실험
- "한 번에 다 섞기"와 "2단계"를 작은 규모로 비교해서 2단계가 실제로 나은지 검증

## 5. 데이터 처리 규칙
1. 모든 데이터를 YOLO 탐지 포맷으로 변환하고 클래스 ID를 4클래스 기준으로 재매핑
2. **split은 영상·시퀀스 단위**로 (프레임 단위로 나누면 비슷한 프레임이 새어 점수가 부풀려짐)
3. 데이터셋을 합치기 전에 **이미지 해시(perceptual hash 포함)로 중복 제거**
4. **누수 제거**: AOD-4에 섞인 Anti-UAV 출신 이미지는 Anti-UAV300 테스트셋과 대조해서 삭제. seraphim도 테스트셋과 대조
5. AOT는 10Hz 영상이라 프레임 간격을 두고 샘플링

## 6. 평가
- **주 지표**: drone 클래스의 AP50, recall
- **보조 지표**: mAP50-95 (작은 박스에 엄격하니 참고용으로만)
- **크기별**: 8×8 미만 / 8~16 / 16~32 / 32px 이상 구간별 AP·recall
- **시점별**: 지상→하늘 테스트와 공중→공중 테스트를 따로 보고
- **오탐률**: 새만 있는 프레임, 빈 하늘 프레임에서 drone 예측 개수
- **속도·크기**: CPU 추론 시간(ms), 파라미터 수, 가중치 용량
- **비교**: YOLO26n vs YOLO26s vs 기존 HF yolov11s 모델

## 7. 확인 필요 (spec 작성 전에 샘플로 확인)
- [ ] DUT Anti-UAV 라벨 포맷 (VOC xml로 추정)
- [ ] Det-Fly 라벨 포맷과 시퀀스 구분 방법
- [ ] AOT 안에서 드론 비중, 클래스 체계, groundtruth 포맷
- [ ] Anti-UAV300의 test split 구성, HF 모델이 쓴 Anti-UAV-RGBT test(91 시퀀스)와 맞출 수 있는지
- [ ] AOD-4 이미지 중 Anti-UAV 출신을 식별할 방법
- [ ] seraphim이 실제로 포함한 원본 데이터셋 목록

## 8. 라이선스
- **YOLO26**: AGPL-3.0. 모델이나 서비스를 공개하면 코드도 공개해야 함 (개인·포트폴리오·오픈소스는 문제없음)
- **CC BY 4.0 데이터 (seraphim, AOD-4)**: README에 출처 표기 필요
- **AOT**: CDLA-Permissive 1.0. 출처 표기 권장
- **DUT, Det-Fly, Anti-UAV**: 레포 라이선스만 있고 데이터 조건은 명시돼 있지 않음. 상업적 사용 전에 원저자 확인 필요

## 9. 다음 단계
1. 각 데이터셋 샘플을 받아서 7번 항목 확인
2. 프로젝트에서 `spec.md` 작성 (폴더 구조, 변환 규칙, data.yaml, 학습 설정, 평가 스크립트)
3. `CLAUDE.md`에 5번 데이터 처리 규칙과 "테스트셋 사용 금지" 규칙 넣기
4. workflow 작성: 다운로드 → 변환 → 중복·누수 제거 → 1단계 → 2단계 → 평가 → n/s 비교
5. 학습은 사용자가 `uv run scripts/train.py --model {n|s} --stage {1|2}` 한 줄로 시작. 얼리스탑, 전체 진행 바, 전원이 꺼져도 이어서 학습하는 안전장치 포함 (spec.md 6절)
