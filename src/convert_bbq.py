"""
BBQ 데이터셋을 대회 포맷으로 변환하는 스크립트.

BBQ (Bias Benchmark for QA) 원본:
    context, question, ans0, ans1, ans2, label, context_condition, question_polarity

대회 포맷:
    sample_id, image_path, context, question, answers(JSON), label

이미지가 없는 텍스트 전용 데이터셋이므로 회색 placeholder 이미지를 생성한다.
파인튜닝 목적은 이미지 이해가 아니라 "근거 없으면 모름 선택" 패턴 학습이다.

실행:
    python src/convert_bbq.py --output-dir ./bbq_data
    python src/convert_bbq.py --output-dir ./bbq_data --max-samples 20000
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
    "Sexual_orientation",
]


def create_placeholder_image(path: Path, size: tuple = (224, 224)) -> None:
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def load_bbq_category(category: str) -> list[dict]:
    """단일 BBQ 카테고리를 로드하여 행 리스트로 반환."""
    try:
        ds = load_dataset("heegyu/bbq", category, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] {category} 로드 실패: {e}")
        return []

    rows = []
    for split in ds:
        for ex in ds[split]:
            rows.append({
                "context":           ex["context"],
                "question":          ex["question"],
                "ans0":              ex["ans0"],
                "ans1":              ex["ans1"],
                "ans2":              ex["ans2"],
                "label":             int(ex["label"]),
                "category":          category,
                "context_condition": ex.get("context_condition", ""),
                "question_polarity": ex.get("question_polarity", ""),
            })
    return rows


def convert(
    output_dir: str,
    categories: list[str],
    max_samples: int | None,
    val_ratio: float,
    seed: int,
) -> None:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. 전체 카테고리 로드
    all_rows = []
    for cat in tqdm(categories, desc="BBQ 카테고리 로드"):
        all_rows.extend(load_bbq_category(cat))

    if not all_rows:
        raise RuntimeError("로드된 데이터가 없습니다. 인터넷 연결을 확인하세요.")

    df_raw = pd.DataFrame(all_rows)

    # 2. 샘플 수 제한 (ambig/disambig 비율 유지)
    if max_samples is not None:
        df_raw = (
            df_raw
            .groupby("context_condition", group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), max_samples // 2),
                random_state=seed,
            ))
            .reset_index(drop=True)
        )

    # 3. 대회 포맷 변환 + placeholder 이미지 생성
    records = []
    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="이미지 생성 및 변환"):
        img_name = f"bbq_img_{idx:06d}.jpg"
        create_placeholder_image(images_dir / img_name)

        records.append({
            "sample_id":  f"BBQ_{idx:06d}",
            "image_path": f"./images/{img_name}",
            "context":    row["context"],
            "question":   row["question"],
            "answers":    json.dumps([row["ans0"], row["ans1"], row["ans2"]]),
            "label":      row["label"],
            # 분석용 메타데이터 (학습 시에는 무시)
            "category":          row["category"],
            "context_condition": row["context_condition"],
            "question_polarity": row["question_polarity"],
        })

    df = pd.DataFrame(records)

    # 4. Train / Val 분리
    val_df   = df.sample(frac=val_ratio, random_state=seed)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)

    # 5. 통계 출력
    print(f"\n{'='*50}")
    print(f"총 샘플 수  : {len(df):,}")
    print(f"Train       : {len(train_df):,}")
    print(f"Val         : {len(val_df):,}")
    print(f"\n[label 분포]")
    print(df["label"].value_counts().to_string())
    print(f"\n[context_condition 분포]")
    print(df["context_condition"].value_counts().to_string())
    print(f"\n[카테고리 분포]")
    print(df["category"].value_counts().to_string())
    print(f"\n저장 위치: {output_dir.resolve()}")
    print("="*50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BBQ → 대회 포맷 변환")
    parser.add_argument("--output-dir",   type=str, default="./bbq_data")
    parser.add_argument("--max-samples",  type=int, default=None,
                        help="전체 샘플 수 제한 (None=전체, ambig/disambig 비율 유지)")
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--categories",   nargs="+", default=BBQ_CATEGORIES,
                        help="사용할 BBQ 카테고리 (기본: 전체)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        output_dir=args.output_dir,
        categories=args.categories,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
