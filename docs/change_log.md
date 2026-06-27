# Change Log

## v8 — 실패 (미제출)
- 파일: `inference_vllm_v8.py`
- 모델: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (LoRA 없음)
- 실패 원인 1: `max_tokens=64` 부족 → reason이 잘리면서 answer_id 미출력 → 파싱 실패 5,369개
- 실패 원인 2 (수정 후): 32B AWQ 추론 시간 ~3~4시간 → 2차 평가 70분 제한 초과 (3~4배)
- 교훈: 32B 모델은 A6000 48GB에서 2차 평가 시간 조건 불충족. max_tokens는 예상 출력 길이 여유 있게 설정 필요

---

## v7 — 0.98825 (v4보다 하락)
- 파일: `inference_vllm_v7.py`
- 변경 1: 프롬프트 개선 (증거 기준 명시 / 고정관념 금지 명시)
- 변경 2: Pass 2 추가 — "모름" 예측 샘플만 재검토
- 결과: 0.9884167 → 0.98825 (소폭 하락)
- 추정 원인:
  - Pass 2가 ambig 샘플(정답이 "모름")을 틀린 특정인으로 바꿈 → ambig 정확도 하락
  - BBQ LoRA가 기존 프롬프트 형식에 최적화되어 있어 프롬프트 변경 시 효과 감소
- 교훈: BBQ LoRA와 프롬프트는 강하게 결합되어 있음. 프롬프트 변경은 LoRA 없이 또는 재학습 후 적용해야 함

---

## v6 — 실패 (미제출)
- 어댑터: `teddykwj/qwen-bbq-lora-v2` (FFN 추가 학습)
- 파싱 실패: 8,500개 / 8,500개 → 전부 label 2
- 원인 1: vLLM `max_lora_rank` 기본값(16) < v2 rank(32) → `max_lora_rank=32` 추가로 해결
- 원인 2: target_modules에 `gate_proj/up_proj/down_proj` 추가 시 visual encoder 레이어까지 학습됨 (동일 이름 공유)
- 원인 3: FFN LoRA 과적합으로 모델 출력 완전 붕괴 (`'!!!!!!!'` 반복 출력)
- 교훈: FFN 레이어 LoRA는 과적합 위험 높음. visual encoder 제외 필수 (regex target_modules 사용)

---

## v5 — 0.9884166667
- 파일: `inference_vllm.py`
- v4 대비 변경: max_pixels `768×28×28` → `1280×28×28`
- 결과: v4와 동일 → 이미지 해상도가 성능에 영향 없음
- 교훈: 텍스트 context만으로 충분히 판단 가능, 이미지는 보조적 역할

---

## v4 — 0.9884166667 ★ 현재 최고
- 파일: `inference_vllm.py`
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` + BBQ LoRA (`teddykwj/qwen-bbq-lora`)
- 추론 엔진: **vLLM** (HuggingFace generate() → vLLM 교체)
- 양자화: fp16 (dtype="float16")
- max_model_len: 8192
- 이미지 토큰 제한: min_pixels=256×28×28 / max_pixels=768×28×28
- 소요 시간: ~34분 (기존 3.7시간 → 6배 빠름)
- 변경 이유: 2차 평가 70분 제한 대응 + 이미지 해상도 정규화
- 파싱 실패: 확인 필요

---

## v3 — 0.9864166667
- 파일: `inference_b.py`
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` + BBQ LoRA (`teddykwj/qwen-bbq-lora`)
- 양자화: **없음 (fp16)**
- torch_dtype: float16
- BATCH_SIZE: 4, max_new_tokens: 32
- 변경 이유: 4-bit 양자화 오차 제거 시 성능 향상 여부 확인
- VRAM: ~14GB
- 결과: v1(0.98808)보다 소폭 하락 → 4-bit 양자화 오차가 성능에 미치는 영향 미미

---

## 실험 A — 실패 (미제출)
- 파일: `inference_a.py`
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` (LoRA 없음)
- 양자화: 4-bit (nf4)
- 결과: 파싱 실패 6,453개 / 8,500개 (76%) → label 2 쏠림 (7,088개)
- 교훈: BBQ LoRA가 JSON 출력 형식 준수에 핵심 역할. 없으면 base 모델이 포맷 불이행

---

## 취소 실험 (72B)
- 모델: `Qwen/Qwen2.5-VL-72B-Instruct` (4-bit 양자화, LoRA 없음)
- 취소 이유: A6000 48GB + 70분 제한 조건에서 불가능 (weights 36GB + 추론 시간 초과)

---

## v2 — 0.8510
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` + BBQ LoRA (`teddykwj/qwen-bbq-lora`)
- 프롬프트: 상세 SYSTEM_PROMPT 추가 (증거 기준 명시, Evidence/Answer 출력 형식)
- BATCH_SIZE: 4, max_new_tokens: 64
- 결과: 파싱 실패 136개 → 점수 하락
- 교훈: 복잡한 프롬프트가 7B 모델에는 역효과. 단순 프롬프트가 우수

---

## v1 — 0.98808
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` + BBQ LoRA (`teddykwj/qwen-bbq-lora`)
- 양자화: 4-bit (nf4, double quant)
- 프롬프트: 단일 user 메시지, JSON 출력 (`{"reason": "...", "answer_id": "0"|"1"|"2"}`)
- BATCH_SIZE: 4, max_new_tokens: 32
- 데이터: test 8,500개
- 파싱 실패: 0개
- label 분포: 0→2919 / 1→2759 / 2→2822
