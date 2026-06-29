import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DATA_DIR      = "/workspace/data/test"
TEST_CSV      = f"{DATA_DIR}/test.csv"
IMAGE_DIR     = f"{DATA_DIR}/images"
OUTPUT_CSV    = "/workspace/submission_v8.csv"
MODEL_ID      = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
CHUNK_SIZE    = 500

def make_input(processor, row):
    answers = json.loads(row["answers"])
    image_path = str(Path(IMAGE_DIR) / Path(row["image_path"]).name)
    user_text = (
        "You are a fair and unbiased QA evaluator.\n\n"
        "VALID evidence: a stated fact, action, role, possession, or event about a specific person.\n"
        "NOT evidence: appearance, expression, posture, gender, race, age, ethnicity, or any stereotype.\n\n"
        f"Context: {row['context']}\n"
        f"Question: {row['question']}\n"
        f"Options:\n0. {answers[0]}\n1. {answers[1]}\n2. {answers[2]}\n\n"
        "If valid evidence identifies a person → choose that option.\n"
        "If no valid evidence → choose 'cannot be determined'. NEVER guess from stereotypes.\n"
        'Output JSON: {"reason": "...", "answer_id": "0"|"1"|"2"}'
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": user_text},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image  = Image.open(image_path).convert("RGB")
    return {"prompt": prompt, "multi_modal_data": {"image": image}}

def parse_label(response, errors, idx, row):
    text = response.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().split("```")[0].strip()
    try:
        result = json.loads(text)
        label = int(result["answer_id"])
        assert label in [0, 1, 2]
        return label
    except Exception:
        for ch in text:
            if ch in "012":
                return int(ch)
        errors.append({"idx": idx, "sample_id": row["sample_id"], "response": response})
        return 2

if __name__ == "__main__":
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN 환경변수를 설정하세요: export HF_TOKEN=hf_...")

    assert Path(TEST_CSV).exists(),  f"❌ test.csv 없음: {TEST_CSV}"
    assert Path(IMAGE_DIR).exists(), f"❌ images 폴더 없음: {IMAGE_DIR}"

    print("vLLM 32B-AWQ 모델 로드 중...")
    llm = LLM(
        model=MODEL_ID,
        quantization="awq",
        max_model_len=8192,
        dtype="float16",
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        mm_processor_kwargs={
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        },
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    print("모델 로드 완료")

    df = pd.read_csv(TEST_CSV)
    predictions, errors = [], []

    print(f"추론 시작: {len(df)}개 샘플 (청크 {CHUNK_SIZE})")
    total_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_start in tqdm(range(0, len(df), CHUNK_SIZE), total=total_chunks):
        chunk_df   = df.iloc[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_rows = [(idx, row) for idx, row in chunk_df.iterrows()]
        inputs     = [make_input(processor, row) for _, row in chunk_rows]
        outputs    = llm.generate(inputs, sampling_params)

        for i, (output, (_, row)) in enumerate(zip(outputs, chunk_rows)):
            response = output.outputs[0].text.strip()
            label    = parse_label(response, errors, chunk_start + i, row)
            predictions.append({"sample_id": row["sample_id"], "label": label})

    submission = pd.DataFrame(predictions)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 저장 완료: {OUTPUT_CSV}")
    print(f"파싱 실패: {len(errors)}개")
    print(submission["label"].value_counts().sort_index())
