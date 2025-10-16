# Intel Arc Graphics + Intel AI Boost NPU로 AI Model 학습 및 추론  


## 0. GPU/NPU 드라이버 준비

GPU Driver 중 운영체제에 맞는 GPU Driver 설치  

- Ubuntu or Windows  
! **WSL2 환경에서는  Ubuntu용 GPU 드라이버 설치**
- [Intel GPU Driver - Ubuntu](https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest)

- [Intel GPU Driver - Windows](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html)

(Intel 클라이언트 GPU용 Intel Deep Learning Essentials 설치)

- PyTorch를 source에서 빌드하지 않고 pip 등을 활용해 바이너리로 설치하는 경우는 Deep Learning Essentials 설치 생략 ([PyTorch Guide 참고](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html))

- [Intel Deep Learning Essentials for GPU](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-9.html)  
사이트에 접속해보면 GPU 용 Deep Learning Essentials 설치 가이드 있음  
! **WSL2 환경에서는 Ubuntu용 설치 방법 사용**

추론용 NPU 드라이버 설치.

- [Intel NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html?utm_source=chatgpt.com)에서 NPU driver 설치


---

## 1. uv 가상환경 복제 및 GPU Test

명령어를 통해 python 3.12 dependency 설치

```bash
uv sync

uv pip install torch --index-url https://download.pytorch.org/whl/xpu
```
Arc GPU 인식 테스트

```bash
uv run test/torch_xpu_test.py
```