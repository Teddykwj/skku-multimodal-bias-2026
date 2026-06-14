# 공부 노트

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

| GPU | VRAM | 용도 |
|-----|------|------|
| RTX 3090 | 24GB | 베이스라인 실행 환경 |
| Kaggle T4 | 16GB | 무료 사용 가능 |
| Colab Pro A100 | 40GB | 유료, 안정적 |
| RTX 4060 (일반 게이밍) | 8GB | AI 학습에는 빡빡 |

**모델 크기별 필요 VRAM (대략)**

```
0.5B 모델 (16-bit) : ~1GB
7B   모델 (16-bit) : ~14GB
7B   모델 (4-bit)  : ~7GB   ← QLoRA 사용 시
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

**LoRA 적용 위치 (target modules)**

주로 Transformer의 Attention 레이어에 적용:
- `q_proj` (query)
- `v_proj` (value)
- 추가로 `k_proj`, `o_proj`, `gate_proj` 등에도 적용 가능

---

## LoRA Rank

추가 레이어의 **크기(복잡도)**를 결정하는 숫자. 클수록 표현력이 높아지지만 VRAM을 더 사용한다.

| rank | 학습 파라미터 | 표현력 | 적합한 태스크 |
|------|--------------|--------|---------------|
| 4  | 매우 적음 | 낮음 | 단순 스타일 변환 |
| 8  | 적음 | 보통 | 일반적인 파인튜닝 |
| **16** | 보통 | **좋음** | **이 대회 추천** |
| 64 | 많음 | 높음 | 복잡한 도메인 적응 |

이 대회처럼 "근거 없으면 모름 선택"이라는 **단순한 패턴 학습**은 rank 16으로 충분하다.

---

## QLoRA

**Q**uantization + **LoRA**의 조합. 4-bit 양자화로 원본 모델을 압축한 뒤 LoRA를 적용한다.

```
[기본 LoRA]
원본 모델 전체 — 16bit로 VRAM에 올림 (~14GB for 7B)
+ 작은 LoRA 레이어 — 이것만 학습

[QLoRA]
원본 모델 전체 — 4bit로 압축해서 VRAM에 올림 (~7GB for 7B)  ← 차이점
+ 작은 LoRA 레이어 — 이것만 학습 (16bit 유지)
```

**성능 비교**

| 방법 | 성능 | 필요 VRAM |
|------|------|-----------|
| Full Fine-tuning | 100% 기준 | 80GB+ |
| LoRA (16-bit) | ~99% | 20GB+ |
| QLoRA (4-bit) | ~97-98% | **12-14GB** |

성능 차이는 1~3% 수준으로 작다. 특히 이 대회처럼 3지선다 분류 태스크에서는 차이가 더욱 미미하다.

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

---

## 이 대회의 핵심 평가 포인트

1. **Answerable 케이스**: context의 명확한 근거를 읽고 올바른 선택지 선택
2. **Ambiguous 케이스**: 인종/성별/민족 정보만으로 추론하지 않고 "모름" 선택

일반 VLM은 Ambiguous 케이스에서 스테레오타입 방향으로 답하는 경향이 있어,
이를 교정하는 것이 성능 향상의 핵심이다.

---

## 모델 선택: Qwen2.5-VL-7B-Instruct

| 항목 | 내용 |
|------|------|
| 파라미터 | 7B |
| 특징 | 지시 준수(instruction following) 능력 강함 |
| QLoRA 환경 | Kaggle T4 16GB에서 학습 가능 |
| 베이스라인 대비 | 0.5B → 7B, 14배 큰 모델 |

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

---

### 1단계: 4-bit 양자화 설정

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
| `bnb_4bit_use_double_quant` | scale factor 자체도 8-bit로 추가 압축 |

**4-bit 양자화란?**

가중치를 float32(40억 개 표현)에서 4-bit(16개만 표현)로 압축하는 것.

```
비트 수 비교:
float32 → 4 bytes/값,  표현 범위 넓음
float16 → 2 bytes/값
nf4     → 0.5 bytes/값, 딱 16단계 중 하나로 반올림
```

70B 모델 기준 140GB → 35GB로 줄어든다.

**역양자화 (계산 시 float16 복원)**

4-bit로 저장할 때 두 가지를 함께 저장한다:
1. **4-bit 인덱스** — 16개 중 몇 번째인지
2. **scale factor** — 원본 값의 범위 정보 (64개 블록 단위)

계산 시:
```
복원값 = NF4_table[인덱스] × scale  →  float16으로 캐스팅  →  행렬곱
```

복원된 float16은 계산 순간에만 존재하고 바로 버려진다. 항상 VRAM에 올라있는 건 4-bit 버전.

**Double Quantization**

`bnb_4bit_use_double_quant=True` — scale factor(양자화 상수) 자체를 한 번 더 8-bit로 압축.
70B 모델 기준 약 2~3GB 추가 절약.

---

### 2단계: 모델 & 프로세서 로드

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

**Qwen2_5_VLForConditionalGeneration**

이름 분해:
- `Qwen2_5`: Alibaba의 Qwen 2.5 모델
- `VL`: Vision-Language (이미지 + 텍스트 동시 처리)
- `ForConditionalGeneration`: 입력을 조건으로 텍스트를 생성하는 구조

**device_map="auto"**

사용 가능한 GPU/CPU VRAM을 자동 계산해서 레이어를 분산 배치. GPU가 1장이면 거기 전부 올라간다.

**torch_dtype=torch.float16**

모델의 기본 연산 타입. 저장은 4-bit, 연산은 float16으로 역할이 분리된다.

| 타입 | 특징 |
|------|------|
| float16 | 구형 GPU(RTX 20/30)에서도 빠름 |
| bfloat16 | 학습 안정성 높음, A100/H100 최적화 |

**AutoProcessor**

이미지와 텍스트를 모델 입력 텐서로 변환하는 전처리기.
- 텍스트 → 토크나이징 → `input_ids`
- 이미지 → 리사이즈/정규화/패치 분할 → `pixel_values`

모델과 동일한 `MODEL_ID`에서 불러와야 설정이 맞아떨어진다.

---

### 3단계: LoRA 어댑터 부착

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
model.print_trainable_parameters()
```

**prepare_model_for_kbit_training**

4-bit 양자화 모델은 바로 학습할 수 없어서 학습 가능한 상태로 준비해주는 함수.

**LoraConfig 파라미터**

| 파라미터 | 의미 |
|----------|------|
| `r` (rank) | 어댑터 행렬의 랭크. 작을수록 파라미터 수 적음 |
| `lora_alpha` | 학습률 스케일 (보통 r의 2배) |
| `lora_dropout` | 과적합 방지 드롭아웃 |
| `target_modules` | LoRA를 붙일 레이어 이름 목록 |
| `bias="none"` | bias는 학습 안 함 |
| `task_type="CAUSAL_LM"` | 텍스트 생성 태스크 명시 |

**target_modules**

LoRA를 어디에 붙일지 지정. 보통 Attention 레이어의 행렬들:
```
q_proj, k_proj, v_proj, o_proj  (+ FFN의 gate_proj, up_proj, down_proj)
```
레이어 이름 확인법: `for name, _ in model.named_modules(): print(name)`

**LoRA 원리**

```
원본 가중치 W: 동결 (gradient 없음)
LoRA 추가:    W + B×A  →  B, A만 학습

크기 비교 (r=8, dim=4096):
  W:    4096 × 4096 = 16,777,216 파라미터
  B+A:  4096×8 + 8×4096 = 65,536 파라미터  (256배 감소)
```

파인튜닝에 필요한 변화량 ΔW가 저랭크 구조를 가진다는 것이 핵심 가정. 실제로 전체 파인튜닝과 성능 차이가 거의 없다.

**get_peft_model**

원본 가중치를 동결하고 LoRA 어댑터만 학습 가능하게 부착. `print_trainable_parameters()`로 전체의 0.1~1%만 학습함을 확인할 수 있다.

**PEFT란?**

Parameter-Efficient Fine-Tuning의 약자. LoRA 외에도 Prefix Tuning, Prompt Tuning, IA3 등 파라미터를 효율적으로 쓰는 파인튜닝 기법들의 모음 (Hugging Face 라이브러리).

---

### 4단계: 데이터셋 (BBQDataset)

**__init__ — 초기화 & 샘플링**

```python
train_dataset = BBQDataset(TRAIN_CSV, BBQ_DATA_DIR, processor, MAX_SEQ_LEN, max_samples=MAX_TRAIN_SAMPLES)
val_dataset   = BBQDataset(VAL_CSV,   BBQ_DATA_DIR, processor, MAX_SEQ_LEN)
```

- `max_samples`가 있으면 `context_condition`(ambig/disambig) 비율을 유지하며 균등 샘플링
- 샘플링 = 전체 데이터 중 일부만 뽑아서 사용 (빠른 실험용)
- 검증 데이터는 `max_samples` 없이 전체 사용 → 정확한 성능 측정을 위해
- **최종 제출 시 `MAX_TRAIN_SAMPLES = None`으로 설정해서 전체 데이터 사용**

이 시점에서는 CSV 로드와 샘플링만 처리됨. 이미지/텍스트 전처리는 아직 안 함 (lazy loading).

**__getitem__ — 샘플 1개 전처리**

```python
assistant_text = json.dumps({
    "reason":    self._build_reason(row),
    "answer_id": str(label),
})
```

모델이 출력해야 할 정답을 JSON 형식으로 만든다. reason을 함께 생성하게 하는 것은 CoT(Chain of Thought) 방식으로 성능 향상 효과가 있다.

```python
full_text = self.processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
user_text_only = self.processor.apply_chat_template(
    [{"role": "user", "content": user_text}],
    tokenize=False, add_generation_prompt=True
)
```

| 옵션 | 의미 |
|------|------|
| `tokenize=False` | 텍스트 문자열만 반환 (이미지와 함께 나중에 토크나이징하기 위해) |
| `add_generation_prompt=False` | 전체 대화 (정답 포함) |
| `add_generation_prompt=True` | 질문까지만 (`<\|im_start\|>assistant\n` 까지) |

두 버전을 만드는 이유: `user_text_only` 길이로 어디까지가 입력인지 구분해서 정답 부분만 loss 계산하기 위해.

```python
inputs = processor(text=..., images=image, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
```

| 옵션 | 의미 |
|------|------|
| `return_tensors="pt"` | 결과를 PyTorch 텐서로 반환 (GPU 연산 가능) |
| `truncation=True` | max_length 초과 시 뒤에서부터 잘라냄 |

**labels 마스킹**

```python
labels = input_ids.clone()
labels[:n_user] = -100  # user 부분 마스킹
```

PyTorch CrossEntropyLoss는 -100인 위치를 자동으로 무시. assistant 응답 부분만 loss 계산.

```
input_ids: [이미지..., 질문..., <assistant>, reason, answer_id]
labels:    [ -100...,  -100...,    -100,      학습,   학습    ]
```

**언어모델 학습 원리 — 다음 토큰 예측**

언어모델은 항상 "다음 토큰 예측"으로만 학습한다.

```
입력: [A, B, C, D, E]
예측:  [B, C, D, E, ?]  ← 각 위치에서 다음 토큰을 예측
```

질문 부분을 -100으로 마스킹하는 이유: 질문을 보고 정답을 맞추는 능력만 학습시키기 위해.

**__getitem__ 반환값**

| 키 | 의미 |
|----|------|
| `input_ids` | 전체 토큰 ID 시퀀스 |
| `attention_mask` | 실제 토큰 위치 표시 (1=유효, 0=패딩) |
| `labels` | loss 계산용 정답 (-100은 무시) |

---

### 5단계: DataLoader & collate_fn

**샘플 처리 흐름**

```
BBQDataset(...)     →  CSV 로드, 샘플링 (이미지 전처리 X)
    ↓
DataLoader 배치 요청
    ↓
__getitem__(0~3)    →  샘플 1개씩 전처리 (lazy loading)
    ↓  (batch_size=4개 모이면)
collate_fn          →  패딩 추가, 배치 텐서로 묶기
    ↓
모델 입력  shape: (4, seq_len)
```

**패딩이 생기는 시점**

`__getitem__`에서는 패딩 없음. 배치 내 샘플들의 길이가 달라서 `collate_fn`에서 맞춰줌.

```python
pad_id = processor.tokenizer.pad_token_id or 0  # pad 토큰 ID (없으면 0)

input_ids = torch.nn.utils.rnn.pad_sequence(
    [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
)
```

`pad_sequence`는 RNN과 무관한 유틸리티 함수. 길이가 다른 시퀀스를 가장 긴 것에 맞춰 패딩.

`batch_first=True` → 결과 shape이 `(배치크기, 시퀀스길이)`

---

### 6단계: TrainingArguments (학습 설정)

**스텝 vs 에폭**

| 개념 | 의미 |
|------|------|
| 스텝 (step) | 배치 1개를 학습한 것 |
| 에폭 (epoch) | 전체 데이터를 한 번 다 돈 것 |

```
데이터 200개, batch_size=4  →  1 에폭 = 50 스텝
```

**주요 설정 옵션**

| 옵션 | 의미 |
|------|------|
| `learning_rate` | 가중치 업데이트 보폭. `새 가중치 = 기존 - 학습률 × 그래디언트` |
| `warmup_ratio` | 초반 N% 구간을 학습률 0→목표값으로 서서히 증가 (초기 불안정 방지) |
| `lr_scheduler_type="cosine"` | 워밍업 이후 학습률을 코사인 곡선으로 서서히 감소 |
| `gradient_accumulation_steps` | N번 배치 후 한 번 업데이트 (메모리 절약, 큰 배치 효과) |
| `fp16=True` | 연산을 float16으로 수행 (속도 향상, 메모리 절약) |
| `gradient_checkpointing=True` | 중간 결과를 버렸다가 역전파 시 재계산 (속도 -20% 대신 메모리 절약) |

**gradient_accumulation 예시**

```
batch_size=4, GRAD_ACCUM=8
→ 실제 효과 = batch_size 32로 학습한 것과 동일
→ GPU에는 4개씩만 올라가므로 메모리 절약
```

**평가 & 저장 설정**

```python
eval_strategy="steps", eval_steps=200    # 200 스텝마다 검증
save_strategy="steps", save_steps=200    # 200 스텝마다 체크포인트 저장
save_total_limit=2                       # 최신 2개만 유지 (오래된 것 자동 삭제)
```

**체크포인트 용도**

1. 학습 중단 시 재개: `trainer.train(resume_from_checkpoint="./output/checkpoint-400")`
2. 최적 모델 선택: 검증 loss가 가장 낮은 시점의 체크포인트를 최종 모델로 사용
3. `load_best_model_at_end=True` 옵션 시 자동으로 최고 성능 체크포인트 로드
