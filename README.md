# DACON 성균관대학교 멀티모달 AI 챌린지 2026

멀티모달 VLM을 활용한 편향 인식 QA 태스크. 이미지 + 텍스트 컨텍스트를 보고 편향 없이 정답(0/1/2)을 선택.

## 최종 결과

| 구분 | 순위 | Score |
|------|------|-------|
| Public | 82등 | 0.98841 |
| Private | 108등 | 0.68428 |

- 평가 지표: Balanced Accuracy = (ambig 정확도 + disambig 정확도) / 2
- 제출 데이터: test 8,500개

## 접근 방법

### 모델 선택

`Qwen2.5-VL-7B-Instruct`를 베이스 모델로 선택. 이미지 + 텍스트를 동시에 처리하는 멀티모달 VLM이며, 7B 규모로 A6000 48GB에서 fp16으로 로드 가능하고 2차 평가 70분 제한을 충족할 수 있는 최대 크기.

### 학습: BBQ QLoRA 파인튜닝

베이스 모델만으로는 두 가지 문제가 있었음:
1. JSON 출력 형식(`{"reason": "...", "answer_id": "0"|"1"|"2"}`)을 제대로 따르지 않음 → 파싱 실패 76%
2. 편향 회피 기준이 없어 고정관념 기반 추론 가능성 있음

이를 해결하기 위해 BBQ(Bias Benchmark for QA) 데이터셋으로 LoRA 파인튜닝 진행:

- **방법**: QLoRA (4-bit 양자화 + LoRA), Kaggle T4 GPU에서 학습
- **LoRA 설정**: rank=16, target_modules=attention layers only (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **학습 효과**: JSON 형식 준수 + 편향 없이 명시적 근거 기반으로 답변하도록 유도
- **주의**: FFN 레이어(`gate_proj`, `up_proj`, `down_proj`)를 target_modules에 추가하면 visual encoder와 이름 충돌로 과적합 발생 → attention only 유지

학습된 어댑터는 HuggingFace Hub에 업로드하여 추론 시 동적으로 로드.

### 추론 파이프라인

```
test.csv (8,500개) + 이미지
        ↓
프롬프트 구성 (Context / Question / Options / JSON 출력 지시)
        ↓
vLLM + LoRA 어댑터 적용 추론 (CHUNK_SIZE=100, max_tokens=32)
        ↓
JSON 파싱 → answer_id 추출 (0 / 1 / 2)
        ↓
submission.csv
```

**vLLM을 선택한 이유**: HuggingFace `generate()` 방식은 8,500개 추론에 3.7시간 소요. 2차 평가의 70분 제한을 맞추기 위해 vLLM으로 전환 → 34분으로 단축 (6배 향상). vLLM의 PagedAttention과 continuous batching이 핵심.

**프롬프트 설계**: 단순한 단일 user 메시지 구조 유지. LoRA가 특정 프롬프트 형식으로 학습되어 있어, 복잡한 system prompt나 형식 변경 시 오히려 성능 하락 확인 (v2: 0.9884 → 0.851).

**이미지 처리**: `min_pixels=256×28×28`, `max_pixels=768×28×28`로 토큰 수 제한. 이미지 해상도를 높여도 성능 변화 없음을 실험으로 확인 — BBQ 태스크 특성상 텍스트 컨텍스트가 판단의 주요 근거.

## 실험 이력

| 버전 | 점수 | 모델 | 핵심 변경 |
|------|------|------|-----------|
| v1 | 0.98808 | 7B + LoRA (4-bit) | 베이스라인 |
| v2 | 0.85100 | 7B + LoRA | 복잡한 system prompt 추가 → 역효과 |
| v3 | 0.98641 | 7B + LoRA (fp16) | 양자화 제거 → 효과 없음 |
| v4 ★ | **0.98841** | 7B + LoRA (fp16) | vLLM 전환, 34분 완료 |
| v5 | 0.98841 | 7B + LoRA (fp16) | 이미지 해상도 ↑ → 효과 없음 |
| v6 | 실패 | 7B + LoRA-v2 | FFN LoRA 과적합, 출력 붕괴 |
| v7 | 0.98825 | 7B + LoRA | 2-pass 재검토 추가 → 오히려 하락 |
| v8 | 실패 | 32B AWQ | 시간 초과 (3~4시간) |

## 주요 교훈

- **BBQ LoRA가 핵심**: LoRA 없이는 base 모델이 JSON 포맷을 지키지 않음 (파싱 실패 76%)
- **프롬프트와 LoRA는 강결합**: LoRA 학습 시 사용한 프롬프트 형식 유지 필수. 변경 시 성능 하락
- **이미지는 보조적 역할**: 텍스트 컨텍스트만으로 대부분 판단 가능, 해상도 개선 효과 없음
- **FFN LoRA는 위험**: attention only 학습이 안전. FFN 추가 시 과적합으로 출력 붕괴
- **vLLM V1 엔진**: Qwen2.5-VL multimodal은 `VLLM_USE_V1=0` 비활성화 불가, 인스턴스별 IPC 제약 있음

## 리소스

**모델**

| 이름 | 링크 |
|------|------|
| qwen-bbq-lora (v1, 사용) | [HuggingFace](https://huggingface.co/Teddykwj/qwen-bbq-lora) |
| qwen-bbq-lora-v2 (실패) | [HuggingFace](https://huggingface.co/Teddykwj/qwen-bbq-lora-v2) |

**데이터**

- 대회 test 데이터: DACON 성균관대학교 멀티모달 AI 챌린지 2026 제공 (재배포 불가)
- BBQ 데이터셋: Parrish et al. (2022), "BBQ: A Hand-Built Bias Benchmark for Question Answering"

## 환경

- GPU: RTX A6000 48GB (2차 평가 기준)
- CUDA: 12.4 / PyTorch: 2.6.0
- 추론 시간: ~34분 (8,500 샘플, 70분 제한)

## 프로젝트 구조

```
src/
├── inference/   — 추론 스크립트 (v1~v9)
├── train/       — BBQ LoRA 학습 노트북
├── data/        — 데이터 전처리
├── test/        — 테스트 스크립트
└── reference/   — 베이스라인, 공개 코드 참고
docs/
├── change_log.md   — 버전별 실험 상세 기록
├── vast_setup.md   — Vast.ai 인스턴스 설정 가이드
└── study_notes.md
```
