# 샘플 확인 결과 (plan.md 7번)

확인일: 2026-11-20. 원본 위치: `data/raw/` (D:\datasets\yolo26-drone-detection\raw 정션)

## 다운로드 현황
| 데이터셋 | 받은 것 | 용량 | 비고 |
|---|---|---|---|
| seraphim | 전체 (train 75,134 / test 8,349) | 9.1 GB zip | HF |
| DUT Anti-UAV | detection train/val/test (5,200 / 2,600 / 2,200) | 1.4 GB zip | Google Drive |
| Det-Fly | 이미지 13,271 + VOC xml 13,271 | 18.8 GB zip | OneDrive (zip 1만 개 제한 때문에 폴더별로 받음) |
| AOT | groundtruth CSV 전체 + 이미지 15,000장 서브셋 | 0.9 GB + 18 GB | `scripts/download_aot_subset.py`, `data/raw/aot/subset_manifest.csv` |
| AOD-4 | v6 YOLO 포맷 전체 22,516 (train 15,761 / valid 4,514 / test 2,241) | 1.1 GB zip | Roboflow |
| Anti-UAV300 | `Anti-UAV-RGBT.zip` 전체 (train 160 / val 67 / test 91 시퀀스) | 6.0 GB zip | 테스트 전용 |

## 1. DUT Anti-UAV 라벨 포맷 ✅
- **VOC xml** (`<split>/xml/*.xml`), 클래스 이름 `UAV` 1개. 이미지는 `<split>/img/00001.jpg`처럼 번호만 붙어 있음
- **영상/시퀀스 ID가 없음** → 시퀀스 단위 split 불가. 공식 train/val/test split을 그대로 쓰는 것이 현실적
- 해상도는 다양함 (예: 550×412)

## 2. Det-Fly 라벨 포맷과 시퀀스 구분 ✅ (문제 있음)
- **VOC xml**, 클래스 `UAV`, 거의 모든 이미지에 객체 1개 (빈 이미지 1장), 3840×2160
- 폴더 `010`(6,358장), `020`(6,913장)은 사실상 두 묶음. xml의 `<folder>`에 train 8,387 / test 4,884 태그가 있지만 원작자가 프레임 단위로 나눈 것으로 보임
- **시퀀스 정보가 없음**: EXIF 없음, xml `path`에도 시퀀스 이름 없음
- 번호가 시간 순서도 아님: 이웃 번호끼리 배경 차이는 작지만(중앙값 3.4/255) 박스 중심은 중앙값 816px씩 튐. 섞인 순서로 보임
- → 시퀀스 단위 split을 하려면 **배경(장면) 유사도로 묶는 클러스터링**이 필요 (spec에서 결정)

## 3. AOT 드론 비중, 클래스 체계, groundtruth 포맷 ✅ (중요)
- groundtruth: `ImageSets/groundtruth.csv` (part1~3), 프레임별 1행. 컬럼 `flight_id, img_name, frame, id, range_distance_m, is_above_horizon, gt_left/top/right/bottom`. `id`는 `Helicopter1`처럼 클래스명+번호. 이미지는 2448×2048 흑백 PNG
- 전체 497만 프레임, 3,438 비행
- **클래스별 프레임 수**: Helicopter 121만, Airplane 67만, Airborne(종류 불명) 47만, Bird 23만, Flock 1.4만, **Drone 3,662 (비행 7개)**, 빈 프레임 275만
- 드론 박스 크기 중앙값 6px (53%가 8px 미만)
- **결론: AOT는 드론 데이터로는 거의 쓸 수 없음.** 비행 7개를 학습용과 홀드아웃 테스트용으로 나누면 양쪽 다 너무 적음. 공중→공중 학습은 사실상 Det-Fly에 의존
- 받은 서브셋: 드론 3,662 + 새 3,446 + 비행기 3,446 + 헬기 3,446 + 빈 하늘 1,000. `Airborne`/`Flock`이 섞인 프레임은 제외

## 4. Anti-UAV300 test split, HF 모델 test(91 시퀀스)와 맞추기 ✅
- 받은 파일명은 `Anti-UAV-RGBT.zip`, **test가 91 시퀀스** → HF `sapoepsilon/yolov11s-drone-detector`가 쓴 Anti-UAV-RGBT test와 개수가 같음 (같은 세트로 보임)
- 시퀀스마다 `visible.mp4` + `visible.json` (`exist`, `gt_rect` [x, y, w, h]) 그리고 IR. **RGB 프레임을 mp4에서 추출해야 함**
- `label_new/{train,val,test}.json`: 시퀀스별 속성 태그 (TC, OV, FM, SV, 난이도 등)
- zip 안의 `framecut.py`는 실행하지 않음

## 5. AOD-4 이미지 중 Anti-UAV 출신 식별 ✅
- **파일명으로 식별 가능**: `20190925_101846_1_1_000_jpg.rf.<hash>.jpg`처럼 Anti-UAV 시퀀스 이름과 프레임 번호가 그대로 남아 있음
- 5,298장 (train 3,720 / valid 1,051 / test 527). **모두 Anti-UAV train 시퀀스 출신이고 test 91 시퀀스 출신은 0장** (파일명 기준)
- 그래도 이름이 바뀐 사본이 있을 수 있으니 pHash 대조는 그대로 진행
- 그 밖의 출처: **YouTube 프레임 10,182장 (`yt-…`, 45%)**, `V_`/`IR_` 접두어 4,942장 (Svanström 계열 데이터셋으로 추정), Unsplash 등 기타 2,094장
- **클래스 순서가 다름**: `0 airplane, 1 bird, 2 drone, 3 helicopter` → 우리 체계(`0 drone, 1 bird, 2 airplane, 3 helicopter`)로 재매핑 필요
- 박스 수는 클래스마다 약 7,900개로 균형, 빈 라벨 파일 596개. 1024×1024로 늘려서(stretch) 리사이즈된 상태

## 6. seraphim이 포함한 원본 데이터셋 ✅ (문제 있음)
- README 기준 23개: Kaggle 3개 (drone-dataset-uav, drone-yolo-detection, fixed-wing-uav), Roboflow Universe 19개, **HF `pathikg/drone-detection-dataset`**
- **plan에서 제외한 `pathikg`(유튜브 프레임, 저작권 애매)가 들어 있음.** 원본 출처 태그가 없어서 그 이미지만 골라 빼기 어려움
- Kaggle drone-yolo-detection, Drone Dataset (UAV)도 들어 있음 → plan에서 별도 제외한 판단과 일치
- 단일 클래스 `0 drone`, 640×640 패딩. 객체 수 train 77,617 / test 8,627, 16px 미만 박스 7.2%
- 실사, 홍보용, 합성 이미지가 섞여 있다고 README에 명시됨. Roboflow 원본에 Anti-UAV 프레임이 섞였을 수 있으니 테스트셋 대조 필수

## spec 작성 전에 결정할 것
1. **공중→공중 테스트셋**: AOT 드론 비행 7개(3,662프레임, 대부분 8px 미만)로 할지, Det-Fly 장면 클러스터 일부로 바꿀지
2. **Det-Fly split 방식**: 장면 클러스터링으로 시퀀스를 대신할지, 공식 태그(train/test)를 쓸지
3. **DUT split**: 시퀀스 ID가 없으니 공식 split을 쓸지
4. **저작권 문제가 있는 데이터**: seraphim 안의 pathikg 이미지, AOD-4의 YouTube 프레임 45%를 어떻게 처리할지 (plan의 pathikg 제외 기준과 충돌)
5. **AOT의 역할**: 드론 데이터가 아니라 오탐(새·비행기·헬기)과 배경 소스로 쓰는 것으로 다시 정의

→ 2026-11-20 추천안대로 확정. 결과는 `spec.md` 0절 (D1~D6)
