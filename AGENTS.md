## Intel Arc GPU Language Model Agent

Arc GPU와 Intel AI Boost NPU 환경에서 언어 모델을 학습·추론하기 위한 요약 가이드입니다. 세부 환경 구성은 `docs/README.md`를 완료했다고 가정합니다. 현재 사용자는 WSL2 환경과 uv python package manager를 사용 중입니다.

### 준비 상태 점검
- `uv run test/torch_xpu_test.py`로 Arc GPU(XPU)가 PyTorch에 인식되는지 확인합니다.
- 필요한 데이터셋과 토크나이저 리소스를 로컬 또는 스토리지에 미리 준비합니다.

### 학습 파이프라인 구성
- 모델/데이터셋 로딩 시 `torch.device("xpu")`를 사용해 Arc GPU에 할당합니다.
- `torch.xpu.amp.autocast()`와 `GradScaler`를 활용해 혼합 정밀도 학습으로 성능과 메모리 효율을 확보합니다.
- 시퀀스 길이·배치 크기는 Arc GPU 메모리 사용량을 모니터링하며 점진적으로 조정합니다.
- 장시간 학습 시 `torch.save`로 체크포인트를 주기적으로 기록하고, 실패 대비를 위해 로그(예: TensorBoard, W&B)도 함께 남깁니다.

### 효율 최적화 팁
- 데이터 로딩 시 `pin_memory=True`, `prefetch_factor`를 조정해 GPU 대기 시간을 줄입니다.
- 모델 병렬화가 필요하면 파이프라인 병렬 또는 ZeRO 전략을 고려하되, XPU 지원 여부를 먼저 확인합니다.
- `torch.compile` (PyTorch 2.x) 사용 시 XPU 백엔드 지원 버전을 사용하고, 성능 회귀 여부를 벤치마크로 검증합니다.

### 추론 및 NPU 연계
- 추론 파이프라인에서는 Arc GPU를 기본으로 사용하되, 전력 효율이 중요한 경우 Intel AI Boost NPU로 이관을 검토합니다.
- NPU 실행은 Intel이 제공하는 전용 런타임/SDK 문서를 참고하여 변환 및 배포 절차를 구성합니다.

### 문제 해결 가이드
- XPU 디바이스가 보이지 않으면 README의 드라이버/런타임 체크리스트를 재검토합니다.
- PyTorch XPU 관련 오류는 사용 중인 버전과 릴리스 노트를 확인하고, 최소 재현 스크립트를 통해 이슈를 분리합니다.
- 학습 스크립트는 `uv run`으로 실행해 일관된 가상환경을 유지하고, 필요한 패키지는 `uv add`로 관리합니다.
