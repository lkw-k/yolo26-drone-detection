# yolo26-drone-detection

YOLO26n / YOLO26s 기반 드론 탐지 모델. **지상→하늘**과 **공중→공중** 시나리오를 하나의 모델로 처리하는 것이 목표입니다.

🤗 **Hugging Face 모델**: [YOLO26n](https://huggingface.co/illimax/yolo26n-drone) · [YOLO26s](https://huggingface.co/illimax/yolo26s-drone)  
🎮 **데모**: [YOLO26n Space](https://huggingface.co/spaces/illimax/yolo26n-drone) · [YOLO26s Space](https://huggingface.co/spaces/illimax/yolo26s-drone)

> ✅ 진행 상태: **학습 완료**. YOLO26n·YOLO26s 가중치와 데모를 Hugging Face에 공개했습니다. 성능은 [결과](#결과) 참고.

## 개요
- **모델**: YOLO26n, YOLO26s (NMS-free). s는 이 레포 파이프라인, n은 이전 파이프라인으로 학습 (데이터 구성이 달라 직접 비교 불가)
- **학습**: 2단계 (seraphim 사전학습 @640 → 실제 시점 혼합 데이터 본학습 @1280)
- **클래스**: drone, bird, airplane, helicopter (추론 시 drone만 사용)
- **데이터**: DUT Anti-UAV, Det-Fly, AOT, AOD-4, seraphim
- **테스트**: Anti-UAV300 RGB (지상→하늘), Det-Fly 홀드아웃 장면 (공중→공중), AOT 드론 비행 (공중→공중 보조), AOT 오탐 비행 (오탐률)

설계 결정과 그 이유는 [docs/design-decisions.md](docs/design-decisions.md)에 있습니다. 구현 명세는 [spec.md](spec.md), 원래 계획은 [plan.md](plan.md)입니다.

## 결과

수치는 모두 **drone 클래스** 기준입니다. ONNX 가중치는 NMS를 따로 적용해야 합니다.

| 모델 | 파라미터 | 입력 | 추천 conf | 가중치 |
|---|---|---|---|---|
| [YOLO26s](https://huggingface.co/illimax/yolo26s-drone) | 9.9M | 1280 | 0.40 | .pt 20.4 MB / .onnx 38.7 MB |
| [YOLO26n](https://huggingface.co/illimax/yolo26n-drone) | 2.5M | 1280 | 0.26 | .pt 5.6 MB / .onnx 10.4 MB |

### YOLO26s (conf 0.40)
| 평가셋 | Precision | Recall | F1 | AP50 |
|---|---|---|---|---|
| Val (9,350장) | 0.936 | 0.849 | 0.891 | 0.893 |
| Anti-UAV300 test (지상→하늘) | 0.956 | 0.655 | 0.777 | 0.801 |
| DUT Anti-UAV test (지상→하늘, 참고용) | 0.989 | 0.957 | 0.973 | 0.977 |
| Det-Fly 테스트 장면 (공중→공중) | 0.869 | 0.552 | 0.675 | 0.633 |
| AOT 홀드아웃 (공중→공중) | - | 0.000 | - | 0.000 |

- **오탐**: AOT 새·비행기·헬기·빈 하늘 1,626프레임에서 drone 예측 0건
- **약점**: 16~32px 드론 (Anti-UAV300 AP50 0.345), 복잡한 공중→공중 배경 (Det-Fly). AOT 홀드아웃 드론은 전부 16px 미만이고 AOT 학습 split에는 드론이 없어서, 일부를 찾아도 airplane으로 분류해 recall이 0
- **속도**: 데스크톱 CPU(PyTorch), 1280 입력 기준 이미지당 약 125ms
- **학습**: seraphim @640 80 epoch → 혼합 데이터 @1280 100 epoch (best: epoch 93)
- 추천 conf는 검증셋에서 drone F1이 가장 높은 지점이며, 테스트 데이터는 학습·임계값 선택에 쓰지 않았습니다.

### YOLO26n (conf 0.26)
| 평가셋 | Precision | Recall | F1 | AP50 |
|---|---|---|---|---|
| Val (4,595장) | 0.926 | 0.842 | 0.882 | 0.877 |
| Anti-UAV300 test (지상→하늘) | 0.891 | 0.390 | 0.543 | 0.520 |
| AOT 홀드아웃 (공중→공중) | 0.904 | 0.171 | 0.287 | 0.181 |

- **오탐**: 새·비행기·헬기·빈 하늘 프레임에서 0~0.1%
- **약점**: recall. 건물 배경, 짙은 안개, 야간, 8px 미만 드론을 자주 놓침
- **속도**: 4스레드 CPU, 1080p 프레임당 약 108ms
- **학습**: seraphim @640 → 혼합 데이터 @1280 100 epoch (best: epoch 97)

> n과 s는 학습 데이터 구성과 검증셋이 달라 수치를 직접 비교할 수 없습니다.

### 사용법
```python
from ultralytics import YOLO

model = YOLO("yolo26s-drone-1280.pt")  # n: "yolo26n-drone-1280.pt", conf=0.26
results = model.predict("image.jpg", imgsz=1280, conf=0.40, classes=[0])  # drone만
```

## 파이프라인
```mermaid
flowchart LR
  A[원본 6종] --> B[YOLO 변환<br/>클래스 4개로 재매핑]
  B --> C[Det-Fly 장면 그룹<br/>20개 장소]
  C --> D[md5 + pHash 해시]
  D --> E[테스트 누수 제거<br/>pathikg 제외<br/>중복 제거 + NCC 검증]
  E --> F[split 생성<br/>테스트 겹침 0 검사]
  F --> G[1단계 학습<br/>seraphim @640]
  G --> H[2단계 학습<br/>혼합 @1280]
  H --> I[평가<br/>크기별·시점별·오탐률]
```

## 데이터 규모 (중복·누수 제거 후)
| 목록 | 이미지 수 | 구성 |
|---|---|---|
| stage1 train / val | 35,426 / 4,024 | seraphim |
| stage2 train | 38,250 | DUT 5,192 + Det-Fly 9,223 + AOT 7,893 + AOD-4 5,942 + seraphim 10,000 |
| stage2 val | 9,350 | DUT 2,597 + Det-Fly 2,007 + AOT 1,688 + AOD-4 1,058 + seraphim 2,000 |
| test_antiuav | 17,078 | Anti-UAV300 test 91 시퀀스, 5프레임마다 1장 |
| test_detfly | 1,990 | Det-Fly 테스트 장면 5개 |
| test_aot_drone | 3,681 | AOT 드론 비행 7개 |
| test_fa | 1,698 | AOT 오탐 테스트 비행 |
| test_dut | 2,200 | DUT test (참고용, 공식 split 오염) |

train/val 119,221장 중 44,171장을 제거했습니다(테스트 누수, 저작권이 애매한 pathikg 사본, 중복). 남은 학습 이미지와 테스트 전용 이미지 사이의 md5·pHash 일치는 0입니다.

## 실행
```bash
uv sync
# 데이터 준비 (원본은 data/raw/)
uv run scripts/convert_<source>.py      # seraphim, dut, detfly, aot, aod4, antiuav_test
uv run scripts/hash_images.py
uv run scripts/dedupe_leak.py
uv run scripts/build_splits.py
# 학습: 명령 하나, early stopping, 크래시·정전 후 같은 명령으로 이어하기
uv run scripts/train.py --model n --stage 1
uv run scripts/train.py --model s --stage 1
uv run scripts/train.py --model n --stage 2
uv run scripts/train.py --model s --stage 2
# 평가
uv run scripts/eval.py --weights data/runs/stage2-yolo26n/weights/best.pt
uv run scripts/eval.py --baseline   # data/weights/baseline_yolov11s.pt 를 직접 받아 둘 것
```

## 레포 구성
```
.
├── plan.md                 # 프로젝트 계획
├── spec.md                 # 구현 명세 (폴더, 변환 규칙, 학습·평가 설정)
├── docs/design-decisions.md  # 설계 결정과 이유
├── notes/sample-check.md   # 데이터셋 샘플 확인 결과
├── configs/                # 1·2단계 학습 설정 (n/s 공통)
├── scripts/                # 변환, 해시, 중복·누수 제거, split, 학습, 평가
├── CLAUDE.md               # Claude Code 프로젝트 규칙
├── .claude/agents/         # Claude Code 서브에이전트 (dataset-auditor, eval-analyst 등)
├── THIRD_PARTY_NOTICES.md
└── LICENSE                 # AGPL-3.0
```
데이터, 가중치, 학습 결과(`data/`)는 레포에 포함하지 않습니다.

### 공개 모델별 학습 코드
| 모델 | 학습 코드 | 비고 |
|---|---|---|
| [illimax/yolo26s-drone](https://huggingface.co/illimax/yolo26s-drone) | 위 파이프라인 (`scripts/train.py`, `scripts/eval.py`) | 2단계 서버 설정: `configs/train_stage2_server.yaml` (batch 30, GPU 3장) |
| [illimax/yolo26n-drone](https://huggingface.co/illimax/yolo26n-drone) | 이전 서버 파이프라인 (`scripts/prepare_*.py`, `make_stage*_lists.py`, `train_stage1.py`, `train_stage2.py`, `eval_test.py`, `ydet/`, `configs/stage*`) | 데이터 구성이 달라 s와 직접 비교 불가 |

## 라이선스
- 코드: [AGPL-3.0](LICENSE) (Ultralytics YOLO26 라이선스를 따름)
- `.claude/agents/`의 일부 에이전트: MIT, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 참고
- 데이터셋은 각 원본의 라이선스를 따르며 이 레포에 포함하지 않습니다.
  - seraphim, AOD-4: CC BY 4.0 / AOT: CDLA-Permissive 1.0
  - DUT Anti-UAV, Det-Fly, Anti-UAV300: 데이터 이용 조건이 명시돼 있지 않아 연구·포트폴리오 용도로만 사용
