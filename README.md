# yolo26-drone-detection

YOLO26n / YOLO26s 기반 드론 탐지 모델. **지상→하늘**과 **공중→공중** 시나리오를 하나의 모델로 처리하는 것이 목표입니다.

> 🚧 진행 상태: 계획 단계 (데이터셋 확정 완료, spec 작성 전)

## 개요
- **모델**: YOLO26n, YOLO26s (NMS-free). 같은 데이터, 같은 설정으로 학습해 비교
- **학습**: 2단계 (seraphim 사전학습 @640 → 혼합 데이터 본학습 @1280)
- **클래스**: drone, bird, airplane, helicopter (추론 시 drone만 사용)
- **데이터**: DUT Anti-UAV, Det-Fly, AOT, AOD-4, seraphim
- **테스트**: Anti-UAV300 RGB (지상→하늘), AOT 홀드아웃 시퀀스 (공중→공중)

자세한 내용은 [plan.md](plan.md)를 참고하세요.

## 레포 구성
```
.
├── plan.md                 # 프로젝트 계획 (결정 사항의 기준 문서)
├── CLAUDE.md               # Claude Code 프로젝트 규칙
├── .claude/agents/         # Claude Code 서브에이전트
│   ├── dataset-auditor.md  # 데이터 누수·중복·split 검사
│   ├── eval-analyst.md     # 크기별·시점별 평가, 모델 비교
│   ├── ml-engineer.md      # ┐
│   ├── mlops-engineer.md   # │
│   ├── data-scientist.md   # │ wshobson/agents (MIT)
│   ├── python-pro.md       # │
│   ├── code-reviewer.md    # │
│   └── debugger.md         # ┘
├── THIRD_PARTY_NOTICES.md
└── LICENSE                 # AGPL-3.0
```

## 라이선스
- 코드: [AGPL-3.0](LICENSE) (Ultralytics YOLO26 라이선스를 따름)
- `.claude/agents/`의 일부 에이전트: MIT, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 참고
- 데이터셋은 각 원본의 라이선스를 따르며 이 레포에 포함하지 않습니다.
