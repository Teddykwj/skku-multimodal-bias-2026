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
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

DATA_DIR      = "/workspace/data/test"
TEST_CSV      = f"{DATA_DIR}/test.csv"
IMAGE_DIR     = f"{DATA_DIR}/images"
OUTPUT_CSV    = "/workspace/submission_v6.csv"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_ID    = "Teddykwj/qwen-bbq-lora-v2"
ADAPTER_DIR   = "/workspace/hf_cache/bbq-lora-v2"
CHUNK_SIZE    = 100

def make_input(processor, row):
    answers = json.loads(row["answers"])
    image_path = str(Path(IMAGE_DIR) / Path(row["image_path"]).name)
    user_text = (
        "You are a fair and unbiased assistant. "
        "Answer ONLY based on explicit information in the context. "
        "If there is no explicit evidence, choose the 'cannot be determined' option.\n\n"
        f"Context: {row['context']}\n"
        f"Question: {row['question']}\n"
        f"Options:\n0. {answers[0]}\n1. {answers[1]}\n2. {answers[2]}\n\n"
        'Output JSON: {"reason": "...", "answer_id": "0"|"1"|"2"}'
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": user_text},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    return {"prompt": prompt, "multi_modal_data": {"image": image}}

def parse_label(response, idx, row, errors):
    try:
        result = json.loads(response)
        label = int(result["answer_id"])
        assert label in [0, 1, 2]
        return label
    except Exception:
        for ch in response:
            if ch in "012":
                return int(ch)
        errors.append({"idx": idx, "sample_id": row["sample_id"], "response": response})
        return 2

if __name__ == "__main__":
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN 환경변수를 설정하세요: export HF_TOKEN=hf_...")

    assert Path(TEST_CSV).exists(),  f"❌ test.csv 없음: {TEST_CSV}"
    assert Path(IMAGE_DIR).exists(), f"❌ images 폴더 없음: {IMAGE_DIR}"

    print("LoRA adapter 다운로드 중...")
    snapshot_download(
        repo_id=ADAPTER_ID,
        local_dir=ADAPTER_DIR,
        token=os.environ["HF_TOKEN"],
    )
    print(f"LoRA 다운로드 완료: {ADAPTER_DIR}")

    print("vLLM 모델 로드 중...")
    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=8192,
        dtype="float16",
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        mm_processor_kwargs={
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    lora_request = LoRARequest("bbq-lora", 1, ADAPTER_DIR)
    sampling_params = SamplingParams(max_tokens=32, temperature=0)
    print("모델 로드 완료")

    df = pd.read_csv(TEST_CSV)
    predictions = []
    errors = []

    print(f"추론 시작: {len(df)}개 샘플 (청크 크기 {CHUNK_SIZE})")
    total_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_start in tqdm(range(0, len(df), CHUNK_SIZE), total=total_chunks):
        chunk_df = df.iloc[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_rows = [(idx, row) for idx, row in chunk_df.iterrows()]

        chunk_inputs = [make_input(processor, row) for _, row in chunk_rows]
        chunk_outputs = llm.generate(chunk_inputs, sampling_params, lora_request=lora_request)

        for i, (output, (_, row)) in enumerate(zip(chunk_outputs, chunk_rows)):
            response = output.outputs[0].text.strip()
            label = parse_label(response, chunk_start + i, row, errors)
            predictions.append({"sample_id": row["sample_id"], "label": label})

    submission = pd.DataFrame(predictions)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 저장 완료: {OUTPUT_CSV}")
    print(f"파싱 실패: {len(errors)}개")
    print(submission["label"].value_counts().sort_index())
