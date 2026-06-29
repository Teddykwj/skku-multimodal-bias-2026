# Vast.ai 인스턴스 설정 가이드

## 인스턴스 종류

| 용도 | GPU | VRAM | 비고 |
|------|-----|------|------|
| 추론 (모델 실행) | - | 48GB | 32B AWQ, 7B+LoRA |
| 검증 (BBQ val) | - | 24GB | 7B+LoRA only |

---

## 1. 추론 인스턴스 (A6000 48GB)

### 접속
```bash
ssh -p <PORT> root@<IP> -L 8080:localhost:8080
```

### 환경 변수
```bash
export HF_TOKEN="<HF_TOKEN>"
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export VLLM_USE_V1=0
```

### 라이브러리 설치
```bash
pip install vllm transformers huggingface_hub accelerate qwen-vl-utils pandas tqdm pillow kaggle
```

### 데이터 다운로드 (Kaggle)
```bash
mkdir -p ~/.kaggle
echo '{"username":"teddykwj","key":"<KAGGLE_API_KEY>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

mkdir -p /workspace/data
cd /workspace/data
kaggle datasets download -d teddykwj/skku-multimodal-bias-2026-data --unzip
```

### 스크립트 업로드 (로컬에서)
```bash
scp -P <PORT> \
    /Users/teddy/projects/dacon/skku-multimodal-bias-2026/src/inference/inference_vllm.py \
    root@<IP>:/workspace/
```

### 추론 실행
```bash
tmux new -s inference
python /workspace/inference_vllm.py
```

### 재접속
```bash
ssh -p <PORT> root@<IP>
tmux attach
```

### 결과 다운로드 (로컬에서)
```bash
scp -P <PORT> \
    root@<IP>:/workspace/submission_v4.csv \
    /Users/teddy/projects/dacon/skku-multimodal-bias-2026/data/
```

---

## 2. 검증 인스턴스 (RTX 4090 24GB)

### 인스턴스 스펙
- Template: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`
- GPU: RTX 4090 (24GB)
- Disk: 50GB 이상

### 접속
```bash
ssh -p <PORT> root@<IP> -L 8080:localhost:8080
```

### 환경 변수
```bash
export HF_TOKEN=<HF_TOKEN>
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export VLLM_USE_V1=0
```

### 라이브러리 설치
```bash
pip install vllm transformers huggingface_hub kaggle pandas tqdm pillow -q
```

### BBQ val 데이터 다운로드
```bash
mkdir -p ~/.kaggle
echo '{"username":"teddykwj","key":"<KAGGLE_API_KEY>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download teddykwj/skku-bbq-data -p /workspace/ --unzip
# → /workspace/bbq_data/val.csv 생성됨
```

### 스크립트 업로드 (로컬에서)
```bash
scp -P <PORT> \
    /Users/teddy/projects/dacon/skku-multimodal-bias-2026/src/inference/inference_val.py \
    root@<IP>:/workspace/
```

### 검증 실행
```bash
python /workspace/inference_val.py
```

### 결과 다운로드 (로컬에서)
```bash
scp -P <PORT> \
    root@<IP>:/workspace/val_errors.csv \
    /Users/teddy/projects/dacon/skku-multimodal-bias-2026/data/
```

---

## 공통 팁

- 장시간 추론은 반드시 `tmux`로 실행 (접속 끊겨도 유지됨)
- `VLLM_USE_V1=0`: Qwen2.5-VL multimodal 사용 시 필수
- `HF_HUB_DISABLE_XET=1`: XET 전송 비활성화 (속도 개선)
- 인스턴스 삭제 전 결과 파일 반드시 로컬로 다운로드
