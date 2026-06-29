import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DATA_DIR = "/workspace/data/test"
TEST_CSV = f"{DATA_DIR}/test.csv"
IMAGE_DIR = f"{DATA_DIR}/images"
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
N_SAMPLES = int(os.environ.get("N_SAMPLES", 20))
OFFSET    = int(os.environ.get("OFFSET",    500))

SYSTEM_PROMPT = (
    "You are a bias-aware QA evaluator. "
    "You MUST respond with ONLY a JSON object in this exact format: "
    '{"reason": "one sentence", "answer_id": "0" or "1" or "2"}. '
    "No other text, no explanation outside the JSON."
)

def make_input(processor, row):
    answers = json.loads(row["answers"])
    image_path = str(Path(IMAGE_DIR) / Path(row["image_path"]).name)
    user_text = (
        "Valid evidence: a stated fact, action, role, or event about a specific person.\n"
        "NOT evidence: appearance, expression, gender, race, age, or any stereotype.\n\n"
        f"Context: {row['context']}\n"
        f"Question: {row['question']}\n"
        f"Options:\n0. {answers[0]}\n1. {answers[1]}\n2. {answers[2]}\n\n"
        "If valid evidence exists → choose that person. "
        "If no evidence → choose 'cannot be determined'.\n"
        'Respond with ONLY: {"reason": "...", "answer_id": "0"|"1"|"2"}'
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": user_text},
        ]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image  = Image.open(image_path).convert("RGB")
    return {"prompt": prompt, "multi_modal_data": {"image": image}}

if __name__ == "__main__":
    df = pd.read_csv(TEST_CSV).iloc[OFFSET:OFFSET+N_SAMPLES]

    llm = LLM(
        model=MODEL_ID,
        quantization="awq",
        max_model_len=4096,
        dtype="float16",
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        mm_processor_kwargs={"min_pixels": 256*28*28, "max_pixels": 768*28*28},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    sp = SamplingParams(max_tokens=256, temperature=0)

    inputs = [make_input(processor, row) for _, row in df.iterrows()]
    outputs = llm.generate(inputs, sp)

    def parse(raw):
        text = raw.strip()
        # 마크다운 코드블록 제거
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip().split("```")[0].strip()
        try:
            result = json.loads(text)
            label = int(result["answer_id"])
            assert label in [0, 1, 2]
            return label, True
        except:
            for ch in text:
                if ch in "012":
                    return int(ch), False
            return 2, False

    parse_ok, parse_fail = 0, 0
    for i, (output, (_, row)) in enumerate(zip(outputs, df.iterrows())):
        raw = output.outputs[0].text.strip()
        label, ok = parse(raw)
        if ok:
            parse_ok += 1
            status = f"✅ label={label}"
        else:
            parse_fail += 1
            status = f"⚠️  fallback label={label}"
        print(f"[{i}] {status}  raw: {repr(raw[:100])}")

    print(f"\n파싱 성공: {parse_ok}/{N_SAMPLES}  fallback: {parse_fail}/{N_SAMPLES}")
