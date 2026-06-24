# 공부 노트

---

## 목차

1. [Kaggle 세팅 방법](#kaggle-세팅-방법)
2. [VRAM](#vram)
3. [양자화 (Quantization)](#양자화-quantization)
4. [LoRA / QLoRA](#lora-low-rank-adaptation)
5. [BBQ 데이터셋](#bbq-데이터셋)
6. [모델 선택](#모델-선택-qwen25-vl-7b-instruct)
7. [train_qlora.ipynb 코드 분석](#train_qloraipynb-코드-분석)
8. [추론 엔진: HuggingFace vs vLLM](#추론-엔진-huggingface-vs-vllm)
9. [Vast.ai 사용법 및 트러블슈팅](#vastai-사용법-및-트러블슈팅)
10. [실험에서 배운 것들](#실험에서-배운-것들)

---

## Kaggle 세팅 방법

### 1. 계정 생성 및 폰 인증

1. [kaggle.com](https://www.kaggle.com) 회원가입
2. 우측 상단 프로필 → **Settings**
3. **Phone Verification** 완료 → GPU 사용 가능해짐 (필수)

---

### 2. 새 노트북 생성

1. 좌측 메뉴 **Code** → **New Notebook**
2. 우측 사이드바 **Session options** 클릭
3. **Accelerator** → `GPU T4 x2` 선택 (32GB VRAM)
4. **Persistence** → `Files only` 권장 (세션 종료 후 파일 유지)

> GPU 할당량: 무료 계정 주 30시간 제한

---

### 3. 데이터 업로드

**방법 A — Kaggle Dataset으로 업로드 (권장)**

1. 좌측 메뉴 **Datasets** → **New Dataset**
2. 대회 데이터(`data/` 폴더) zip으로 압축 후 업로드
3. 노트북에서 우측 사이드바 **Add data** → 본인 데이터셋 추가
4. `/kaggle/input/데이터셋이름/` 경로로 접근

**방법 B — 노트북 안에서 직접 업로드**
```python
# 소용량 파일만 가능, 대용량은 Dataset 방식 사용
```

---

### 4. HuggingFace 모델 접근 설정

Qwen2.5-VL처럼 HuggingFace에서 모델을 다운로드할 때 토큰이 필요할 수 있다.

1. [huggingface.co](https://huggingface.co) 로그인 → Settings → **Access Tokens**
2. 토큰 생성 (read 권한)
3. Kaggle 노트북에서 **Add-ons** → **Secrets** → `HF_TOKEN` 이름으로 저장
4. 노트북에서 사용:

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
```

---

### 5. 라이브러리 설치

노트북 상단 셀에 추가:

```bash
# QLoRA 파인튜닝에 필요한 패키지
!pip install -q transformers peft bitsandbytes trl accelerate
!pip install -q datasets
```

> Kaggle에는 PyTorch, CUDA가 기본 설치되어 있어 별도 설치 불필요

---

### 6. 세션 제한 및 주의사항

| 항목 | 내용 |
|------|------|
| 최대 세션 시간 | 12시간 |
| 주간 GPU 할당량 | 30시간 |
| 인터넷 연결 | 기본 비활성화 → Settings에서 켜야 HuggingFace 접근 가능 |
| 저장 공간 | `/kaggle/working/` 아래 20GB 제한 |

**인터넷 연결 활성화**: 우측 사이드바 **Session options** → **Internet** → `On`

---

### 7. 모델/결과 저장

세션이 끊기면 `/kaggle/working/` 내용이 사라지므로,
학습 완료 후 반드시 **Output으로 저장**해야 한다.

```python
# 체크포인트를 Output 경로에 저장
model.save_pretrained("/kaggle/working/qwen_qlora_checkpoint")
```

저장된 파일은 노트북 우측 **Output** 탭에서 다운로드 가능.

---

## VRAM

GPU에 내장된 전용 메모리. AI 모델을 GPU에서 실행할 때 **모델 전체가 VRAM에 올라가야** 한다.

```
RAM  : CPU 옆에 있는 일반 메모리 (프로그램, OS 등)
VRAM : GPU 안에 있는 메모리 (모델 가중치, 연산 중간값)
```

VRAM이 부족하면 모델 로드 자체가 안 되거나 OOM(Out of Memory) 에러 발생.

**주요 GPU별 VRAM**

| GPU | VRAM | 특징 |
|-----|------|------|
| Kaggle T4 | 16GB | 무료, QLoRA 학습 가능 |
| RTX 3090 / 4090 | 24GB | 4-bit 추론 여유, fp16은 빠듯 |
| RTX 5090 | 32GB | Blackwell 아키텍처, fp16 추론 여유 |
| A100 | 40~80GB | 데이터센터용, 안정적 |
| RTX A6000 | 48GB | 2차 평가 환경 |

**모델 크기별 필요 VRAM**

```
7B 모델 (fp16, 16-bit) : ~14GB
7B 모델 (4-bit QLoRA)  : ~7GB
7B 모델 (vLLM fp16)    : ~14GB 모델 + KV 캐시 ~8GB = ~22GB
```

---

## 양자화 (Quantization)

모델 가중치를 저장하는 **숫자 정밀도를 줄여** 메모리를 절약하는 기법.

```
16-bit (원본): 3.14159265358979  ← 정밀한 소수점
4-bit  (양자화): 3.1             ← 대략적인 값
```

정밀도를 잃지만, 파라미터 수가 수십억 개인 LLM은 약간의 근사가 성능에 거의 영향을 주지 않는다.

**메모리 절약 효과**

```
16-bit → 4-bit : 메모리 4분의 1로 감소
7B 모델: 14GB → 7GB
```

### BitsAndBytesConfig 상세

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

| 옵션 | 의미 |
|------|------|
| `load_in_4bit` | 모델 가중치를 4-bit로 압축해서 로드 |
| `bnb_4bit_compute_dtype` | 실제 연산은 float16으로 수행 |
| `bnb_4bit_quant_type="nf4"` | NormalFloat4 방식 (QLoRA 논문 제안, int4보다 정밀도 높음) |
| `bnb_4bit_use_double_quant` | scale factor 자체도 8-bit로 추가 압축, ~2GB 추가 절약 |

**역양자화 (계산 시 float16 복원)**

```
복원값 = NF4_table[인덱스] × scale  →  float16으로 캐스팅  →  행렬곱
복원된 float16은 계산 순간에만 존재, 항상 VRAM에 올라있는 건 4-bit 버전
```

---

## LoRA (Low-Rank Adaptation)

LLM을 효율적으로 파인튜닝하는 기법. 원본 모델은 **얼려두고(freeze)**, 작은 추가 레이어만 학습한다.

**핵심 아이디어 — 행렬 분해**

```
원본 가중치 행렬 W (1000 × 1000) = 100만 파라미터  ← 학습 안 함

LoRA 추가:
  행렬 A (1000 × rank) + 행렬 B (rank × 1000)      ← 이것만 학습
  rank=16일 때: 1000×16 + 16×1000 = 32,000 파라미터
```

전체 파라미터의 0.1~1% 수준만 학습하므로 메모리와 시간이 크게 절약된다.

**LoRA Rank**

| rank | 표현력 | 적합한 태스크 |
|------|--------|---------------|
| 4  | 낮음 | 단순 스타일 변환 |
| 8  | 보통 | 일반적인 파인튜닝 |
| **16** | **좋음** | **이 대회 추천** |
| 64 | 높음 | 복잡한 도메인 적응 |

**LoRA 적용 위치 (target modules)**

주로 Transformer의 Attention 레이어에 적용:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- 추가로 `gate_proj`, `up_proj`, `down_proj` (FFN)

---

## QLoRA

**Q**uantization + **LoRA**의 조합. 4-bit 양자화로 원본 모델을 압축한 뒤 LoRA를 적용한다.

```
[QLoRA]
원본 모델 전체 — 4bit로 압축해서 VRAM에 올림 (~7GB for 7B)
+ 작은 LoRA 레이어 — 이것만 학습 (16bit 유지)
```

**성능 비교**

| 방법 | 성능 | 필요 VRAM |
|------|------|-----------|
| Full Fine-tuning | 100% 기준 | 80GB+ |
| LoRA (16-bit) | ~99% | 20GB+ |
| QLoRA (4-bit) | ~97-98% | **12-14GB** |

> 이 대회 실험 결과: LoRA+fp16(0.98641) vs LoRA+4bit(0.98808) — 사실상 동일, 4-bit가 오히려 약간 높았음

**QLoRA 구현에 필요한 라이브러리**
- `bitsandbytes` : 4-bit 양자화
- `peft` : LoRA 어댑터
- `transformers` : 모델 로드/학습
- `trl` : SFT(Supervised Fine-Tuning) 트레이너

---

## BBQ 데이터셋

**Bias Benchmark for QA** (Parrish et al., ACL 2022)

이 대회 데이터의 원본으로 추정되는 텍스트 기반 편향 QA 벤치마크.

- 약 58,000개 QA 쌍
- 9개 편향 카테고리: 나이, 성별, 인종, 종교, 국적, 장애, 성적 지향, 사회경제적 지위, 외모
- 각 문항에 **Ambiguous context** + **Disambiguated context** 두 버전 존재

```
[Disambiguated — 근거 있음]
context: "White 사람이 이야기를 했는데 질문받자 설명이 무너졌다."
→ 정답: White person

[Ambiguous — 근거 없음]
context: "Roma 사람과 White 사람이 있다."
→ 정답: Not enough information  (스테레오타입으로 답하면 틀림)
```

### 이 대회의 핵심 평가 포인트

1. **Disambiguated 케이스**: context의 명확한 근거를 읽고 올바른 선택지 선택
2. **Ambiguous 케이스**: 인종/성별/민족 정보만으로 추론하지 않고 "모름" 선택

평가 지표: `Balanced Accuracy = (Ambiguous Accuracy + Disambiguated Accuracy) / 2`

일반 VLM은 Ambiguous 케이스에서 스테레오타입 방향으로 답하는 경향이 있어, 이를 교정하는 것이 성능 향상의 핵심이다.

---

## 모델 선택: Qwen2.5-VL-7B-Instruct

| 항목 | 내용 |
|------|------|
| 파라미터 | 7B |
| 특징 | 지시 준수(instruction following) 능력 강함 |
| QLoRA 환경 | Kaggle T4 16GB에서 학습 가능 |
| 2차 평가 환경 | RTX A6000 48GB, 70분 제한 |

---

## train_qlora.ipynb 코드 분석

### 전체 흐름

```
1. 모델 로드 (4-bit 양자화)
2. LoRA 어댑터 부착
3. 데이터셋 준비
4. 학습 설정 (TrainingArguments)
5. 학습 실행 (Trainer)
```

### 1단계: 모델 & 프로세서 로드

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

**Qwen2_5_VLForConditionalGeneration** — 이름 분해:
- `Qwen2_5`: Alibaba의 Qwen 2.5 모델
- `VL`: Vision-Language (이미지 + 텍스트 동시 처리)
- `ForConditionalGeneration`: 입력을 조건으로 텍스트를 생성하는 구조

**device_map="auto"** — 사용 가능한 GPU/CPU VRAM을 자동 계산해서 레이어를 분산 배치.

**AutoProcessor** — 이미지와 텍스트를 모델 입력 텐서로 변환하는 전처리기.
- 텍스트 → 토크나이징 → `input_ids`
- 이미지 → 리사이즈/정규화/패치 분할 → `pixel_values`

### 2단계: LoRA 어댑터 부착

```python
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

| 파라미터 | 의미 |
|----------|------|
| `r` (rank) | 어댑터 행렬의 랭크 |
| `lora_alpha` | 학습률 스케일 (보통 r의 2배) |
| `lora_dropout` | 과적합 방지 |
| `bias="none"` | bias는 학습 안 함 |
| `task_type="CAUSAL_LM"` | 텍스트 생성 태스크 명시 |

### 3단계: 데이터셋 (BBQDataset)

**labels 마스킹** — 질문 부분은 학습에서 제외, 정답 부분만 loss 계산:

```python
labels = input_ids.clone()
labels[:n_user] = -100  # user 부분 마스킹 (CrossEntropyLoss가 -100 위치 무시)
```

```
input_ids: [이미지..., 질문..., <assistant>, reason, answer_id]
labels:    [ -100...,  -100...,    -100,      학습,   학습    ]
```

### 4단계: TrainingArguments

| 옵션 | 의미 |
|------|------|
| `learning_rate` | 가중치 업데이트 보폭 |
| `warmup_ratio` | 초반 N% 학습률 0→목표값 (초기 불안정 방지) |
| `lr_scheduler_type="cosine"` | 코사인 곡선으로 서서히 감소 |
| `gradient_accumulation_steps` | N번 배치 후 한 번 업데이트 (메모리 절약) |
| `gradient_checkpointing=True` | 중간 결과 버렸다가 역전파 시 재계산 (속도 -20%, 메모리 절약) |

---

## 추론 엔진: HuggingFace vs vLLM

추론 코드를 작성할 때 **모델을 어떤 방식으로 실행하느냐**가 속도에 큰 영향을 미친다.

### HuggingFace generate() 방식 (기본)

```python
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
model = PeftModel.from_pretrained(model, adapter_id)  # LoRA 부착
output = model.generate(**inputs, max_new_tokens=32)
```

- 직관적이고 설정이 쉬움
- 배치 처리는 수동으로 묶어야 함 (BATCH_SIZE=4 등)
- 8,500개 기준 약 **3.7시간** 소요

### vLLM 방식 (고속 추론 엔진)

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model=BASE_MODEL_ID, enable_lora=True, max_model_len=8192)
outputs = llm.generate(inputs, SamplingParams(max_tokens=32))
```

- **PagedAttention**: KV 캐시를 페이지 단위로 관리 → 메모리 효율 극대화
- **Continuous Batching**: 요청이 끝나는 즉시 다음 요청을 채워 GPU 가동률 극대화
- 8,500개 기준 약 **30~40분** 예상 (5~7배 빠름)
- LoRA는 로컬 경로로 제공해야 함 → 먼저 `snapshot_download()`로 다운로드

**속도 비교 요약**

| 방식 | 8,500개 소요 시간 | 2차 평가 70분 제한 |
|------|------------------|-------------------|
| HuggingFace | ~3.7시간 | ❌ 초과 |
| vLLM | ~30~40분 | ✅ 통과 |

### vLLM 사용 시 주의사항

**1. `if __name__ == '__main__':` 필수**

vLLM은 내부적으로 multiprocessing을 사용한다. 이 가드 없이 실행하면 자식 프로세스가 무한 재생성되어 에러 발생:

```python
# ❌ 잘못된 방식
llm = LLM(...)  # 모듈 레벨에서 직접 실행

# ✅ 올바른 방식
if __name__ == "__main__":
    llm = LLM(...)
```

**2. `max_model_len` 설정**

Qwen2.5-VL은 이미지를 토큰으로 변환하는데, 이미지 해상도에 따라 수백~수천 토큰이 소모된다. 텍스트 + 이미지 토큰 합계가 `max_model_len`을 초과하면 에러:

```
ValueError: The decoder prompt (length 5747) is longer than max_model_len of 4096.
```

→ `max_model_len=8192` 이상으로 설정

**3. os.environ 설정은 import 전에**

```python
import os
os.environ["VLLM_USE_V1"] = "0"   # 반드시 vllm import 전에 설정

from vllm import LLM  # 이 순서를 지켜야 환경변수가 적용됨
```

**4. LoRA는 로컬 경로 필요**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="teddykwj/qwen-bbq-lora", local_dir="/workspace/bbq-lora")

lora_request = LoRARequest("bbq-lora", 1, "/workspace/bbq-lora")
```

---

## Vast.ai 사용법 및 트러블슈팅

GPU를 시간당 요금으로 대여하는 플랫폼. 저렴하지만 불안정한 호스트가 있어 주의 필요.

### 인스턴스 선택 기준

| 항목 | 기준 | 이유 |
|------|------|------|
| VRAM | 24GB+ (fp16 7B) | 모델 14GB + KV 캐시 여유분 |
| DLPerf | 높을수록 좋음 | 실제 딥러닝 추론 속도 지표 |
| 호스트 운영 기간 | 1개월+ | 안정성 지표 |
| Disk | 50GB+ | 모델(~14GB) + 데이터(~5GB) |
| 네트워크 | 1000 Mbps+ | 모델 다운로드 속도 |

> **주의**: Vast.ai 스펙 표기가 부정확한 경우 있음. "48GB RTX 4090" 같은 표기는 실제로 24GB일 수 있음 → `nvidia-smi`로 반드시 확인

### 이미지 선택

| 이미지 | 추천 여부 | 이유 |
|--------|----------|------|
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` | ✅ 추천 | 2차 평가 환경과 동일, 안정적 |
| `vastai/vllm` | ❌ 비추천 | 공식 vLLM 포크 아님, v0.23.0 커스텀 빌드, IPC 에러 발생 |
| `vllm/vllm-openai` | ⚠️ 조건부 | 공식이지만 추가 패키지 설치 필요 |

### 설치 순서 (pytorch 이미지 기준)

```bash
# 1. 패키지 설치
pip install vllm transformers peft qwen-vl-utils pandas tqdm pillow "kaggle>=1.6.0"

# 2. HF 토큰 설정
export HF_TOKEN=hf_...

# 3. Kaggle 설정
mkdir -p /root/.kaggle
cat > /root/.kaggle/kaggle.json << 'EOF'
{"username":"teddykwj","key":"KGAT..."}
EOF
chmod 600 /root/.kaggle/kaggle.json

# 4. 데이터 다운로드
mkdir -p /workspace/data
kaggle competitions download -c <대회명> -p /workspace/data
cd /workspace/data && unzip -q *.zip

# 5. tmux로 백그라운드 실행
tmux new -s infer
python /workspace/inference_vllm.py
# Ctrl+B, D 로 detach (세션 유지)
# tmux attach -t infer 로 재접속
```

### 자주 발생한 에러

**에러 1: `vastai/vllm` IPC 에러**
```
RuntimeError: Engine core initialization failed. no sessions
```
→ 원인: `vastai/vllm` 이미지의 커스텀 vLLM(v0.23.0)이 컨테이너 환경에서 ZMQ IPC 초기화 실패  
→ 해결: `pytorch/pytorch` 이미지로 교체 후 `pip install vllm`

**에러 2: multiprocessing spawn 에러**
```
RuntimeError: An attempt has been made to start a new process before bootstrapping
```
→ 원인: `if __name__ == '__main__':` 가드 없이 vLLM 실행  
→ 해결: 모든 실행 코드를 `if __name__ == '__main__':` 블록 안으로 이동

**에러 3: prompt 길이 초과**
```
ValueError: The decoder prompt (length 5747) is longer than max_model_len of 4096.
```
→ 원인: 이미지 토큰 + 텍스트가 max_model_len 초과  
→ 해결: `max_model_len=8192`로 증가

**에러 4: GPU 메모리 점유 (프로세스 없음)**
```
Free memory (2.81/23.52 GiB) is less than desired utilization (21.17 GiB)
```
→ 원인: 이전 실행 크래시 후 GPU 메모리가 해제되지 않은 orphaned 상태  
→ 해결: Vast.ai 대시보드에서 인스턴스 Stop → Start (재시작)

**에러 5: Kaggle CLI 인증 실패**
```
KeyError: 'username'
```
→ 원인: 구버전 kaggle CLI는 KGAT 토큰 단독 인증 불지원  
→ 해결: kaggle.json에 username + key 직접 작성

### 연결 끊김 대응

SSH 연결이 끊겨도 프로세스 유지를 위해 **tmux** 필수:

```bash
tmux new -s infer        # 새 세션 생성
# 작업 실행...
Ctrl+B, D               # 세션 detach (백그라운드로 전환)

# 재접속 후
tmux attach -t infer    # 세션 재연결
```

---

## 실험에서 배운 것들

### BBQ LoRA의 역할: 편향 교정 + 출력 형식 준수

LoRA 없이 base 모델만 실행했을 때 (실험 A):

```
파싱 실패: 6,453개 / 8,500개 (76%)
label 분포: 0→728 / 1→684 / 2→7,088  ← label 2에 극단적으로 쏠림
```

BBQ LoRA는 단순히 편향 판단만 개선하는 게 아니라, **JSON 형식으로 정확하게 출력하는 능력**도 함께 학습되어 있다. LoRA 없이는 base 모델이 포맷을 지키지 못해 파싱이 거의 다 실패한다.

### 복잡한 프롬프트는 7B 모델에 역효과

상세한 SYSTEM_PROMPT와 Evidence/Answer 형식을 추가했을 때 (v2):

```
파싱 실패: 136개 → 점수: 0.8510  (v1: 0.98808)
```

지시사항이 길어질수록 7B 모델은 형식을 따르지 못하는 경우가 늘어난다. 단순하고 명확한 프롬프트가 훨씬 효과적.

### 양자화 정밀도가 성능에 미치는 영향: 거의 없음

| 버전 | 설정 | 점수 |
|------|------|------|
| v1 | LoRA + 4-bit | **0.98808** |
| v3 | LoRA + fp16 | 0.98641 |

fp16이 오히려 약간 낮았다. 7B 모델 + 단순 분류 태스크에서는 4-bit 양자화 오차가 성능에 거의 영향을 주지 않는다.

### 앙상블 규칙: 단순 다수결 불가

이 대회 규칙:
> "단순 다수결/룰 기반 앙상블은 불가 → LLM이 종합하여 결정해야 함"

여러 추론 결과를 모아 다수결로 최종 답을 정하는 방식은 규칙 위반. LLM이 여러 reasoning을 보고 직접 최종 답을 내야 한다 (Self-Consistency 방식).

### GPU 대여 시 스펙 허위 표기 주의

Vast.ai에서 "48GB" 표기가 실제로는 24GB였던 사례 발생. 인스턴스 접속 후 반드시:

```bash
nvidia-smi  # 실제 VRAM 확인
```
