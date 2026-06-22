import os
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
if "HF_TOKEN" not in os.environ:
    raise EnvironmentError("HF_TOKEN 환경변수를 설정하세요: export HF_TOKEN=hf_...")

DATA_DIR      = "/workspace/data/test"
TEST_CSV      = f"{DATA_DIR}/test.csv"
IMAGE_DIR     = f"{DATA_DIR}/images"
OUTPUT_CSV    = "/workspace/submission_b.csv"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_ID    = "teddykwj/qwen-bbq-lora"
BATCH_SIZE    = 4

assert Path(TEST_CSV).exists(),  f"❌ test.csv 없음: {TEST_CSV}"
assert Path(IMAGE_DIR).exists(), f"❌ images 폴더 없음: {IMAGE_DIR}"

print("모델 로드 중... [실험 B: LoRA 유지 + fp16]")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=os.environ["HF_TOKEN"])
model.eval()
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
processor.tokenizer.padding_side = "left"
print(f"모델 로드 완료 | VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

df = pd.read_csv(TEST_CSV)
predictions = []
errors = []

def make_messages(row):
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
    return [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": user_text},
    ]}]

def parse_label(response, idx, row):
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

print(f"추론 시작: {len(df)}개 샘플, 배치 크기 {BATCH_SIZE}")
total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
for batch_start in tqdm(range(0, len(df), BATCH_SIZE), total=total_batches):
    batch_rows = df.iloc[batch_start:batch_start + BATCH_SIZE]
    batch_messages = [make_messages(row) for _, row in batch_rows.iterrows()]

    batch_texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    all_images = []
    for msgs in batch_messages:
        imgs, _ = process_vision_info(msgs)
        if imgs:
            all_images.extend(imgs)

    inputs = processor(
        text=batch_texts,
        images=all_images if all_images else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    for i, (_, row) in enumerate(batch_rows.iterrows()):
        generated = output_ids[i][input_len:]
        response = processor.decode(generated, skip_special_tokens=True).strip()
        label = parse_label(response, batch_start + i, row)
        predictions.append({"sample_id": row["sample_id"], "label": label})

submission = pd.DataFrame(predictions)
submission.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 저장 완료: {OUTPUT_CSV}")
print(f"파싱 실패: {len(errors)}개")
print(submission["label"].value_counts().sort_index())
