# Change Log

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
