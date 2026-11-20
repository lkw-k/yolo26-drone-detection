# spec: YOLO26n / YOLO26s 드론 탐지

`plan.md`의 결정을 구현 수준으로 구체화한 문서. 근거 데이터는 `notes/sample-check.md`.
작성일: 2026-11-20

## 0. 샘플 확인 후 확정한 결정
| # | 항목 | 결정 | 이유 |
|---|---|---|---|
| D1 | 공중→공중 테스트셋 | **주 테스트: Det-Fly 장면 클러스터 홀드아웃(약 15%)**. 보조 테스트: AOT 드론 비행 7개 전체(3,662프레임, 초소형 드론) | AOT 드론은 7비행·박스 중앙값 6px라 학습/테스트로 나눌 수 없음. 7개 전체를 테스트로만 씀 |
| D2 | Det-Fly split | 배경 유사도로 프레임을 묶어 **장면 그룹**을 만들고, 그룹 단위로 train/val/test = 70/15/15. 원본 train/test 태그는 쓰지 않음 | 원본 태그는 프레임 단위라 누수가 생김 |
| D3 | DUT split | **공식 split 사용**: train→train, val→val, test→보조 평가(지상→하늘 참고용) | 시퀀스 ID가 없음. split 사이 근접 중복은 감사(audit)에서 수치로 보고 |
| D4 | 저작권이 애매한 데이터 | AOD-4: **YouTube 프레임(`yt-*`) 10,182장 제외**. seraphim: **`pathikg/drone-detection-dataset`을 받아 pHash로 대조하고 일치하는 이미지 제외**(대조 후 pathikg는 삭제) | plan의 pathikg 제외 기준을 그대로 따름 |
| D5 | AOT의 역할 | **오탐 대상(새·비행기·헬기)과 빈 하늘 배경 소스**. 비행 단위로 train/val/오탐 테스트 = 70/15/15 | 드론 데이터로는 쓸 수 없음 |
| D6 | AOD-4의 Anti-UAV 출신 이미지 | 5,298장 **전부 제외** | Anti-UAV300 전체가 테스트 전용 데이터셋이라, train 시퀀스 출신이어도 같은 촬영 환경에서 생기는 누수를 피함 |

이 결정에 따라 테스트 전용(학습·검증·튜닝·threshold 선택 금지) 데이터는 다음 4개다.
1. Anti-UAV300 RGB 전체. 평가에는 test 91개 시퀀스만 사용
2. Det-Fly 테스트 장면 그룹
3. AOT 드론 비행 7개
4. AOT 오탐 테스트 비행

## 1. 폴더 구조
```
yolo26-drone-detection/
  configs/
    train_stage1.yaml       # Ultralytics train args (n/s 공통)
    train_stage2.yaml
  scripts/
    download_aot_subset.py  # (기존)
    common.py               # class map, paths, image resize/hardlink, manifest I/O
    convert_seraphim.py
    convert_dut.py
    convert_detfly.py       # includes scene grouping
    convert_aot.py
    convert_aod4.py
    convert_antiuav_test.py
    hash_images.py
    dedupe_leak.py
    build_splits.py
    train.py
    eval.py
  data/  -> D:\datasets\yolo26-drone-detection (정션, git 제외)
    raw/                    # 원본. 읽기 전용, 스크립트가 쓰지 않음
    yolo/<source>/images/<split>/*.jpg
    yolo/<source>/labels/<split>/*.txt
    manifests/<source>.csv  # 변환 결과 (1행 = 이미지 1장)
    manifests/hashes.parquet
    manifests/final.csv     # 중복·누수 제거 후
    reports/                # dedupe/leak/audit 리포트
    splits/*.txt            # Ultralytics 입력용 이미지 경로 목록
    splits/*.yaml           # data.yaml
    runs/                   # 학습 결과 (D 드라이브)
```
- 이미지를 바꾸지 않는 경우(seraphim, DUT, AOD-4)에는 raw에서 **하드링크**로 만든다. raw와 같은 D 드라이브라 용량이 늘지 않는다. 리사이즈가 필요한 경우(Det-Fly, AOT)에는 다시 인코딩한다(JPEG q95).
- **긴 변이 1920px보다 크면 1920으로 축소한다.** 2단계 학습 imgsz가 1280이라 1920이면 스케일 증강 여유가 충분하다. Det-Fly(3840×2160)는 JPEG 디코딩 비용이 8배가량 줄어든다. 라벨은 정규화 좌표라 바뀌지 않는다.

## 2. 매니페스트 형식 (`data/manifests/<source>.csv`)
| 컬럼 | 설명 |
|---|---|
| `image` | `data/yolo/...` 이미지의 절대 경로 |
| `source` | `seraphim`, `dut`, `detfly`, `aot`, `aod4`, `antiuav` |
| `group` | split 단위 ID (영상, 시퀀스, 비행, 장면 그룹). split은 항상 이 값 단위로 나눈다 |
| `split` | `train`, `val`, `test`, `test_aux`, `test_fa` |
| `viewpoint` | `g2a`(지상→하늘), `a2a`(공중→공중), `mixed`(seraphim, AOD-4 기타) |
| `role` | `pos`(드론 있음), `neg`(새·비행기·헬기만 있음), `bg`(빈 이미지) |
| `n_drone`, `n_bird`, `n_airplane`, `n_helicopter` | 박스 수 |
| `w`, `h` | 저장된 이미지 크기 |

## 3. 데이터셋별 변환 규칙
클래스 맵은 고정: `0 drone, 1 bird, 2 airplane, 3 helicopter`. 박스는 이미지 경계로 자르고, 자른 뒤 폭이나 높이가 1px 미만이면 버린다.

### seraphim (`convert_seraphim.py`)
- 이미 YOLO 포맷이고 클래스는 `0` 하나 → 그대로 하드링크
- `group`: 원본 출처 정보가 없으므로 파일명 stem. 1·2단계 학습용일 뿐 평가에는 쓰지 않으므로 허용
- split: 원본 train→`train`, test→`val`
- D4: pathikg 대조는 `dedupe_leak.py`에서 처리 (아래 6절)

### DUT Anti-UAV (`convert_dut.py`)
- VOC xml `<split>/xml/*.xml`, 클래스 `UAV` → `0 drone`
- `group`: `dut_<split>` (시퀀스 정보 없음, D3)
- split: train→`train`, val→`val`, test→`test_aux`
- viewpoint `g2a`

### Det-Fly (`convert_detfly.py`)
- VOC xml, 클래스 `UAV` → `0 drone`. 1920×1080으로 축소
- **장면 그룹 만들기 (D2)**:
  1. 각 이미지를 64×36 흑백 썸네일로 만든다 (드론은 4K에서 작아서 썸네일에 거의 영향이 없음)
  2. 썸네일 평균 절대 차이가 `T` 이하인 이미지 쌍을 연결하고, union-find로 연결 성분을 그룹으로 삼는다. 쌍 비교는 최근접 이웃 검색으로 한다 (전체 쌍 비교 1.3만² 회피)
  3. `T`는 거리 분포가 갈라지는 지점(이웃 거리 히스토그램)에서 고르고, 값을 스크립트 상수와 리포트에 남긴다
  4. 그룹별 대표 이미지 contact sheet(`data/reports/detfly_groups.jpg`)를 만들어 **사람이 눈으로 확인**한다. 너무 큰 그룹(전체의 30% 이상)이 생기면 `T`를 낮춘다
- 그룹 단위로 train/val/test = 70/15/15 (이미지 수 기준, seed 0)
- viewpoint `a2a`. 빈 이미지 1장은 `bg`

### AOT (`convert_aot.py`)
- 입력: `data/raw/aot/subset_manifest.csv` + groundtruth CSV. 이미지는 2448×2048 → 1920×1606
- 클래스 매핑: `Bird`→1, `Airplane`→2, `Helicopter`→3, `Drone`→0. 서브셋에는 `Airborne`/`Flock` 프레임이 없음
- `group`: `flight_id`
- split:
  - 드론 비행 7개 → 전부 `test_aux` (D1, 보조 공중→공중 테스트)
  - 나머지 비행 → 비행 단위 train/val/`test_fa` = 70/15/15 (seed 0). `test_fa`는 오탐률 평가 전용
- role: 드론 프레임 `pos`, 새·비행기·헬기 프레임 `neg`, 빈 프레임 `bg`. viewpoint `a2a`

### AOD-4 (`convert_aod4.py`)
- 클래스 재매핑: `0 airplane→2`, `1 bird→1`, `2 drone→0`, `3 helicopter→3`
- **제외**: `yt-*` (D4), Anti-UAV 출신 `^\d{8}_\d{6}_\d+_\d+_\d+` (D6)
- 남는 것: `V_*` 4,942장(181개 영상, Svanström 계열) + 기타(Unsplash 등) 2,094장 = 7,036장. 박스: 비행기 974 / 새 5,729 / 드론 1,735 / 헬기 2,374
- `group`: `V_*`는 영상 ID(`V_AIRPLANE_0011`), 기타는 `_jpg.rf.` 앞의 원본 이름. Roboflow 증강본이 같은 그룹에 묶이게 하려는 것
- 원본 train/valid/test는 무시하고 그룹 단위로 train/val = 85/15
- viewpoint `mixed`. 빈 라벨 파일은 `bg`

### Anti-UAV300 테스트 (`convert_antiuav_test.py`)
- test 91개 시퀀스의 `visible.mp4`에서 **5프레임마다 1장** 추출 (`STRIDE = 5`)
- `visible.json`의 `exist`가 1이면 `gt_rect [x,y,w,h]`로 `0 drone` 라벨을 만들고, 0이면 빈 라벨(`bg`)
- `group`: 시퀀스 이름. split `test`, viewpoint `g2a`
- 누수 검사용으로 train/val/test 전체 시퀀스의 **모든 프레임** pHash도 따로 계산한다 (이미지는 저장하지 않음)
- zip 안의 `framecut.py`는 쓰지 않는다

## 4. 중복 제거와 누수 제거
### `hash_images.py`
- 모든 매니페스트 이미지와 Anti-UAV 전체 프레임에 대해 md5와 pHash(`imagehash.phash`, 64bit)를 계산 → `hashes.parquet`
- seraphim은 640 패딩이 있으므로 **균일한 테두리를 잘라낸 뒤** pHash를 계산한다 (패딩이 있으면 해시가 달라져 대조가 안 됨). AOD-4의 1024 stretch는 pHash가 32×32로 리사이즈하므로 그대로 둔다

### `dedupe_leak.py`
순서대로 적용하고, 삭제한 이미지는 모두 사유와 함께 `data/reports/removed.csv`에 남긴다.
1. **테스트 누수 (blocking)**: train/val 이미지 중 테스트 전용 데이터(0절 4개 + Anti-UAV train/val 프레임)와 md5가 같거나 pHash 해밍 거리 ≤ 6이면 train/val 쪽을 삭제
2. **pathikg 제외 (D4)**: seraphim 이미지 중 pathikg 이미지와 md5가 같거나 해밍 거리 ≤ 4이면 삭제. 대조가 끝나면 pathikg 원본은 삭제
3. **학습 데이터 사이 중복**: md5 동일 또는 해밍 거리 ≤ 4인 묶음은 1장만 남긴다. 남길 우선순위는 Det-Fly > DUT > AOT > AOD-4 > seraphim (원본에 가까운 쪽). 묶음이 train과 val에 걸치면 val 쪽을 남기고 train 쪽을 지운다
4. 결과 → `manifests/final.csv`, 요약 → `data/reports/dedupe_leak.md`
- 해밍 거리 임계값은 실행 후 경계 근처 쌍 20개를 contact sheet로 보고 조정한다

## 5. split 구성 (`build_splits.py`)
`final.csv`를 읽어 `data/splits/`에 이미지 경로 목록(txt)과 data.yaml을 만든다. 모든 data.yaml의 `names`는 4클래스 고정 → 1단계 가중치를 2단계에서 head까지 그대로 이어받는다.

| 파일 | 내용 |
|---|---|
| `stage1_train.txt` / `stage1_val.txt` | seraphim train / seraphim test (6절 제거 후) |
| `stage2_train.txt` | DUT train + Det-Fly train + AOT train + AOD-4 train + seraphim train 무작위 10,000장(replay, seed 0) |
| `stage2_val.txt` | DUT val + Det-Fly val + AOT val + AOD-4 val + seraphim test 무작위 2,000장 |
| `test_antiuav.txt` | Anti-UAV300 test (지상→하늘 주 테스트) |
| `test_detfly.txt` | Det-Fly test 그룹 (공중→공중 주 테스트) |
| `test_aot_drone.txt` | AOT 드론 비행 7개 (보조, 초소형) |
| `test_dut.txt` | DUT test (보조, 지상→하늘 참고용) |
| `test_fa.txt` | AOT 오탐 테스트 비행 (새·비행기·헬기·빈 하늘) |

- 2단계 예상 규모(제거 전): train 약 4.2만장 = DUT 5.2k + Det-Fly 9.3k + AOT 7.9k + AOD-4 6.0k + seraphim 10k
- 시점 균형: `build_splits.py`가 viewpoint별 이미지 수와 박스 수를 출력한다. g2a와 a2a 중 한쪽 드론 이미지 수가 다른 쪽의 2배를 넘으면 경고만 하고 자동 조정은 하지 않는다 (사람이 결정)
- **테스트 목록이 train/val 목록과 겹치면 스크립트가 실패**한다 (경로, `group` 둘 다 검사)

## 6. 학습
### 실행 방법
```
uv run scripts/train.py --model n --stage 1
uv run scripts/train.py --model s --stage 1
uv run scripts/train.py --model n --stage 2
uv run scripts/train.py --model s --stage 2
uv run scripts/train.py --model n --stage 1 --smoke   # 파이프라인 확인용 소규모 실행
```
- 명령 하나로 학습이 시작된다. 별도 준비 단계가 필요 없다
- run 이름은 `stage{1|2}-yolo26{n|s}`, 결과는 `data/runs/` (D 드라이브)
- 2단계는 같은 모델의 `stage1-yolo26{n|s}/weights/best.pt`에서 시작한다. 없으면 바로 에러로 끝낸다
- `--smoke`: 각 split에서 200장, epochs 1, imgsz 640, run 이름 뒤에 `-smoke`

### 설정 (`configs/train_stage{1,2}.yaml`, n/s 공통)
| 항목 | 1단계 | 2단계 |
|---|---|---|
| 시작 가중치 | `yolo26{n,s}.pt` (COCO) | 1단계 `best.pt` |
| data | `splits/stage1.yaml` | `splits/stage2.yaml` |
| imgsz | 640 | 1280 |
| epochs | 80 | 100 |
| **patience (얼리스탑)** | 15 | 20 |
| batch | 32 | 8 |
| optimizer / lr | auto / 기본값 | auto / 기본값 |
| close_mosaic | 10 | 10 |
| seed / deterministic | 0 / True | 0 / True |
| save_period | 5 | 5 |
| workers | 8 | 8 |
| amp | True | True |
- **얼리스탑**: Ultralytics `patience`를 쓴다. val fitness가 `patience` epoch 동안 나아지지 않으면 멈추고 `best.pt`가 최종 결과가 된다
- **n과 s는 같은 yaml을 쓴다** (hard rule 5). 12GB에서 OOM이 나면 batch를 n/s **둘 다** 같은 값으로 줄이고 yaml을 고친다. 모델별 예외는 두지 않는다

### 진행 상황 표시 (바 형태)
- Ultralytics가 기본으로 보여 주는 epoch 내부 batch 진행 바는 그대로 둔다
- 그 위에 `train.py`가 **전체 학습 진행 바**(tqdm, callback `on_fit_epoch_end`)를 추가한다
  - 표시: `epoch 23/100 [███████░░░] 23% | box 1.12 cls 0.85 | drone mAP50 0.71 (best 0.73 @ep20) | patience 3/20 | GPU 9.1G | ETA 3h12m`
  - 이어서 학습할 때는 시작 epoch부터 바를 채운다
- epoch마다 `results.csv`와 `results.png`(Ultralytics 기본)가 갱신되므로 학습 중에도 곡선을 볼 수 있다

### 컴퓨터가 꺼져도 학습이 사라지지 않게 하는 안전장치
1. **매 epoch 체크포인트**: Ultralytics가 epoch마다 `weights/last.pt`를 저장한다 (optimizer, EMA, epoch 포함). 전원이 꺼져도 최대 1 epoch만 잃는다
2. **손상 대비 백업**: `last.pt`를 쓰는 도중에 꺼지면 파일이 깨질 수 있다. `on_model_save` callback에서 `last.pt`를 `last_safe.pt.tmp`로 복사하고 fsync한 뒤 `os.replace`로 `last_safe.pt`를 교체한다. 교체가 원자적이라 `last_safe.pt`는 항상 온전한 직전 체크포인트다
3. **주기 스냅샷**: `save_period: 5` → 5 epoch마다 `epoch{N}.pt`. 앞의 둘이 다 깨져도 돌아갈 곳이 있다
4. **자동 이어하기**: 같은 명령을 다시 치면 `train.py`가 run 폴더를 확인한다
   - `last.pt`가 있고 학습이 끝나지 않았으면 → `resume=True`로 이어서 학습
   - `last.pt`를 읽을 수 없으면 → `last_safe.pt`, 그다음 가장 최근 `epoch{N}.pt` 순서로 시도하고, 무엇으로 이어하는지 출력
   - 이미 끝난 run이면 → 아무것도 하지 않고 결과 위치만 출력
   - 처음부터 다시 하려면 `--fresh`. 기존 폴더는 지우지 않고 `<name>-old-<timestamp>`로 이름만 바꾼다
   - 이어하기는 run 폴더의 `args.yaml`을 그대로 쓰므로 설정이 중간에 바뀌지 않는다
5. **절전 방지**: 학습하는 동안 Windows `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`로 절전 진입을 막고, 끝나거나 에러가 나면 해제한다. 프로세스 단위 설정이라 시스템 설정은 바꾸지 않는다
6. **재현 정보**: 시작할 때 run 폴더에 `run_info.json`(git commit, split 파일 md5, 명령줄, GPU)을 남긴다
- 정전 자체는 막을 수 없다. 위 장치는 **다시 켠 뒤 같은 명령 한 줄로 이어서 학습**할 수 있게 해 준다

## 7. 평가 (`eval.py`)
```
uv run scripts/eval.py --weights data/runs/stage2-yolo26n/weights/best.pt
uv run scripts/eval.py --baseline     # sapoepsilon/yolov11s-drone-detector
```
- 추론은 Ultralytics `predict`(conf 0.001, 우리 모델은 imgsz 1280, baseline은 원래 imgsz 640)로 하고, 지표는 직접 계산한다
- **매칭**: drone 클래스만 대상, IoU ≥ 0.5, 점수 내림차순 greedy
- **지표**: test 목록별 drone AP50, recall, mAP50-95(참고용)
- **크기 구간**: GT 박스의 `sqrt(w·h)`(저장된 이미지 픽셀 기준) <8 / 8–16 / 16–32 / ≥32. 매칭되지 않은 예측은 자기 크기 구간에 넣는다 (COCO area range 방식)
- **recall과 오탐률의 conf threshold**: stage2 val에서 drone F1이 최대인 conf를 n과 s 각각 구하고, **두 값의 평균 하나**를 모든 모델(baseline 포함)에 똑같이 쓴다. 테스트 데이터로는 정하지 않는다
- **오탐률**: `test_fa.txt`에서 frame당 drone 예측 수. 새만 / 비행기만 / 헬기만 / 빈 하늘로 나눠 보고
- **비용**: 파라미터 수, 가중치 용량, CPU 지연 시간(batch 1, 100장 평균, warmup 10장)
- 출력: `data/runs/eval/<model>.json` + 비교 표 `data/reports/eval_summary.md`. 해석은 `eval-analyst`가 맡는다

## 8. 의존성
- 추가: `uv add imagehash polars` (polars는 지금 ultralytics를 통해 간접으로 설치돼 있음 → 직접 의존성으로 명시)
- 진행 바는 tqdm (ultralytics 의존성)
- 일회성 다운로드 도구(`huggingface_hub`, `gdown`)는 `uvx`로만 실행하고 의존성에 넣지 않는다

## 9. 구현 순서와 확인 방법
각 단계가 끝나면 사용자에게 확인을 받고 다음 단계로 넘어간다.

| # | 작업 | 확인 방법 |
|---|---|---|
| 1 | `common.py` + 변환 스크립트 6개 | 소스별 이미지/라벨 수가 `notes/sample-check.md`와 맞는지, 무작위 20장에 박스를 그린 contact sheet |
| 2 | Det-Fly 장면 그룹 | 그룹 수와 크기 분포, contact sheet를 사용자가 눈으로 확인 |
| 3 | pathikg 받기 → `hash_images.py` → `dedupe_leak.py` | 삭제 사유별 개수, 경계 쌍 contact sheet |
| 4 | `build_splits.py` | 테스트/학습 겹침 0, viewpoint 균형 표 |
| 5 | **`dataset-auditor` 실행** | 모든 항목 PASS |
| 6 | `train.py` + `--smoke` (1단계, 2단계) | 진행 바 표시, smoke 학습 중간에 프로세스를 강제 종료한 뒤 같은 명령으로 이어하기가 되는지, `last.pt`를 일부러 망가뜨렸을 때 `last_safe.pt`로 이어하는지 |
| 7 | `eval.py` + smoke 가중치로 테스트 | 지표 파일 생성, baseline 평가 동작 |
| 8 | 1단계 n → s (사용자가 명령 실행) | `results.csv`, val 곡선 |
| 9 | 2단계 n → s (사용자가 명령 실행) | 같음 |
| 10 | 평가 + **`eval-analyst`** | 비교 표, 다음 단계 제안 |

## 10. 라이선스 메모
- README 출처 표기: seraphim(CC BY 4.0), AOD-4(CC BY 4.0), AOT(CDLA-Permissive 1.0)
- AOD-4 안의 `V_*`(Svanström 계열)와 Unsplash 이미지는 원출처 라이선스를 README에 적는다. 상업적 사용 전에는 다시 확인한다
- DUT, Det-Fly, Anti-UAV는 데이터 이용 조건이 명시돼 있지 않음 → 연구·포트폴리오 용도로만 사용
