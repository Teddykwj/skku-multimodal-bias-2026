# Change Log

## 실험 B (진행 중)
- 파일: `inference_b.py`
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` + BBQ LoRA
- 양자화: **없음 (fp16)**
- 변경 이유: 4-bit 양자화 오차 제거 시 성능 향상 여부 확인
- VRAM 예상: ~14GB
- 점수: 미제출

---

## 실험 A (진행 중)
- 파일: `inference_a.py`
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` (LoRA 없음)
- 양자화: 4-bit (nf4)
- 변경 이유: BBQ LoRA 기여도 확인
- 점수: 미제출

---

## v3 (취소)
- 모델: `Qwen/Qwen2.5-VL-72B-Instruct` (4-bit 양자화, LoRA 없음)
- BATCH_SIZE: 1
- 취소 이유: A6000 48GB + 70분 제한 조건에서 불가능 (weights 36GB + 추론 시간 초과)

---

## v2 — 0.8510
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` + BBQ LoRA (`teddykwj/qwen-bbq-lora`)
- 프롬프트: 상세 SYSTEM_PROMPT 추가 (증거 기준 명시, Evidence/Answer 출력 형식)
- BATCH_SIZE: 4, max_new_tokens: 64
- 결과: 파싱 실패 136개 → 점수 하락
- 교훈: 복잡한 프롬프트가 7B 모델에는 오히려 역효과. 단순 프롬프트가 우수

---

## v1 — 0.98808
- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` + BBQ LoRA (`teddykwj/qwen-bbq-lora`)
- 양자화: 4-bit (nf4, double quant)
- 프롬프트: 단일 user 메시지, JSON 출력 (`{"reason": "...", "answer_id": "0"|"1"|"2"}`)
- BATCH_SIZE: 4, max_new_tokens: 32
- 데이터: test 8,500개
- 파싱 실패: 0개
- label 분포: 0→2919 / 1→2759 / 2→2822
