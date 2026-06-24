# 2026 성균관대학교 멀티모달 AI Bias 챌린지 — 프로젝트 계획

## 대회 개요

| 항목 | 내용 |
|------|------|
| 주최 | 성균관대 지능형멀티미디어연구센터 |
| 기간 | 2026.06.01 ~ 2026.06.29 10:00 |
| 평가 | **Balanced Accuracy** (ambig/disambig 각각 Accuracy → 평균) |
| 상금 | 720만 원 |
| 태스크 | 이미지 + 텍스트 기반 3지선다 VQA → 정답 인덱스(0/1/2) 예측 |
| 일일 제출 한도 | 5회 |

---

## 문제 설명

이미지와 상황 설명(context)이 주어졌을 때, 성별·인종·민족 등 사회적 편향이 포함된 질문에 대해 올바른 선택지를 예측하는 멀티모달 QA 태스크.

핵심 난이도는 두 가지 케이스를 정확히 구분하는 것:

- **Disambiguated**: context에 명확한 근거가 있는 경우 → 근거에 기반한 답 선택
- **Ambiguous**: 판단에 필요한 정보가 부족한 경우 → "Cannot be determined" 선택

> **중요**: ambig/disambig 여부는 테스트셋에서 공개되지 않음.
> Balanced Accuracy = (ambig Accuracy + disambig Accuracy) / 2 이므로 두 케이스 모두 잘해야 함.

---

## 평가 구조

| 구분 | 비율 | 용도 |
|------|------|------|
| Public Score | 테스트셋 60% | 리더보드 실시간 확인 |
| Private Score | 테스트셋 40% | 1차 평가 (100%) |

- **2차 평가**: Private 상위 15팀(예비 5팀 포함) 대상, Hidden 데이터셋 + 코드 검증
- 최종 수상: 오프라인 시상식 (2026.07.14) 참석 필수

### 2차 평가 정량 환산식

```
점수 = 배점 × (내 점수 / 최고 점수)^N
```
- N: 조정 계수 (1~5 사이, 비공개)

---

## 대회 규칙 (핵심)

### ❌ 사용 불가
- **외부 API 모델 금지**: OpenAI API, Gemini API, HuggingFace Inference API 등 원격 추론 일체 불가
- 모델 가중치를 직접 로드하여 실행하는 방식만 허용
- 2026.06.01 이후 공개된 모델 불가 (2026.05.31 이전 공개 모델만 사용)

### ✅ 사용 가능
- 오픈소스 모델 직접 로드 (Qwen2.5-VL-7B 등)
- BBQ 등 외부 데이터 (저작권 준수)
- 앙상블, 다중 프롬프트 (단, 최종 답변은 LLM이 생성해야 함)
- 단순 다수결/룰 기반 앙상블은 불가 → LLM이 종합하여 결정해야 함

### ⚠️ Data Leakage 주의
- 테스트셋 문항 패턴을 분석해 유사 학습 데이터 생성 금지

---

## 데이터 구조

```
data/
├── train/
│   ├── images/train_img_0000.jpg   ← 샘플 1개만 제공
│   └── train.csv
├── test/
│   ├── images/                     ← 8,500개
│   └── test.csv
└── sample_submission.csv
```

### CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| sample_id | 샘플 ID |
| image_path | 이미지 상대 경로 |
| context | 상황 설명 텍스트 |
| question | 질문 |
| answers | 선택지 3개 (JSON 배열) |
| label | 정답 인덱스 0/1/2 (train만) |

---

## 핵심 인사이트

이 대회의 데이터는 **BBQ (Bias Benchmark for QA)** 데이터셋 구조와 동일하다.
BBQ는 약 58,000개의 QA 쌍으로 구성되며, 각 문항에 대해 ambiguous / disambiguated 두 버전을 제공한다.
BBQ 데이터로 파인튜닝하면 "모름" 판단 경계를 직접 학습할 수 있다.

---

## 전략

### 모델
- **베이스 모델**: `Qwen2.5-VL-7B-Instruct`
- 2026.05.31 이전 공개 → 규칙 준수
- Kaggle T4 16GB에서 QLoRA 파인튜닝 가능
- 2차 평가 환경(A6000 48GB)에서 더 큰 모델도 검토 가능

### 학습 방법
- **QLoRA** (4-bit 양자화 + LoRA)
- VRAM 약 11~12GB → Kaggle T4 16GB 내 처리 가능
- LoRA rank: 16, target modules: q/k/v/o_proj

### 학습 데이터
- BBQ 데이터셋 (GitHub: `nyu-mll/BBQ`)
- Train: 52,643개 / Val: 5,849개
- Ambiguous 50% + Disambiguated 50% 비율 유지

### 추론 전략
- Step 1: 이미지 시각적 설명 추출
- Step 2: (이미지 설명 + context) 기반 편향 없는 판단
- 목표 속도: 샘플당 0.5초 이내 (8,500개 기준 약 70분)

---

## 추론 시간 제약

| 평가셋 | 샘플 수 | 권장 시간 |
|--------|---------|-----------|
| Test | 8,500개 | 70분 이내 |
| Hidden | 1,500개 | 13분 이내 |

- 베이스라인: 8,500개 약 13분 (vLLM 0.5B 기준)
- 7B 모델은 vLLM 사용 시 충분히 가능할 것으로 예상

---

## 작업 계획

### Step 1 — 데이터 준비
- [x] BBQ 데이터셋 다운로드 및 분석
- [x] 대회 포맷으로 변환 스크립트 작성 (`convert_bbq.ipynb`)
- [x] 학습/검증 데이터 분리 (Train 52,643 / Val 5,849)
- [x] Kaggle Dataset 업로드 완료

### Step 2 — 파인튜닝 (Kaggle)
- [x] `Qwen2.5-VL-7B-Instruct` QLoRA 학습 코드 작성 (`train_qlora.ipynb`)
- [ ] Kaggle T4 x2에서 학습 실행
- [ ] 체크포인트 저장 및 검증셋 성능 확인

### Step 3 — 추론
- [ ] 개선된 추론 코드 작성 (2-step 프롬프트 + vLLM)
- [ ] 파인튜닝 모델로 test 8,500개 추론
- [ ] 제출 파일(`submission.csv`) 생성 및 제출

### Step 4 — 개선
- [ ] Balanced Accuracy 기준 ambig/disambig 각각 성능 분석
- [ ] 앙상블 또는 self-consistency 적용 (LLM 기반으로 구현)
- [ ] 2차 평가 대비 코드 정리 및 문서화

---

## 개선된 프롬프트 방향

```
Step 1 - Describe the people in the image:
(Describe their appearance, gender, race, age if visible)

Step 2 - Analyze the context:
Context: {context}
Determine whether the context provides EXPLICIT evidence.

Step 3 - Answer:
You are a fair and unbiased assistant.
Answer ONLY based on explicit information in the context.
If no explicit evidence → choose "cannot be determined".
Do NOT use visual appearance or stereotypes.

Output JSON: {"reason": "...", "answer_id": "0"|"1"|"2"}
```

---

## 환경

| 항목 | 내용 |
|------|------|
| 학습 환경 | Kaggle (T4 16GB × 2) |
| 로컬 환경 | GPU 없음 (코드 작성 전용) |
| 2차 평가 환경 | RTX A6000 48GB, Python 3.10, CUDA 12.4, PyTorch 2.6.0 |
| 프레임워크 | HuggingFace Transformers, PEFT, bitsandbytes |
| 추론 엔진 | vLLM |

---

## 2차 평가 제출 자료 (상위 15팀 해당)

메일 제목: `[팀명] 2026 성균관대학교 멀티모달 AI 챌린지 최종 산출물 제출`  
제출처: dacon@dacon.io

### ① 코드 / 모델 / 외부데이터 [필수]

- 학습 코드(train)와 추론 코드(inference) **반드시 분리**하여 별도 파일로 구성
- 파일 형태: `.py` 또는 `.ipynb`
- 외부 데이터 사용 시 해당 데이터 파일 포함
- 제출한 코드는 **오차 범위 내 Private Score 복원** 가능해야 함
- 코드/주석 인코딩: **UTF-8**
- 모든 코드는 오류 없이 실행되어야 함 (라이브러리 로딩 포함)
- 개발 환경(OS) 및 라이브러리 버전 기재

### ② 솔루션 발표 자료 [필수]

- 발표 15분 분량, 자유 양식, **PDF로 제출** (PPT 불가)
- 시상식 당일 상위 3팀이 해당 PDF로 발표

### ③ 참가 자격 증빙 서류 [필수]

- 대회 기간(6.1~6.29) 기준 재학생 또는 휴학생 (졸업유예생 불가)
- 재학증명서 / 휴학증명서 등 팀원 전체 제출

---

## 일정

| 날짜 | 내용 |
|------|------|
| 06.29 (월) 10:00 | 대회 종료 / 최종 제출 마감 |
| 06.29 (월) 12:00 | 2차 평가 대상자 산출물 제출 시작 |
| 07.02 (목) 10:00 | 산출물 제출 마감 (dacon@dacon.io) |
| 07.10 (금) | 오프라인 시상식 대상자 안내 |
| 07.14 (화) | 오프라인 시상식 |
