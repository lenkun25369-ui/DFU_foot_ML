#!/usr/bin/env python3
"""依 evaluate 產生的 predictions.csv，把原始影像複製到 TP/TN/FP/FN。

只讀取已儲存的 label 與 prediction，不重新預測或重新設定門檻。
Python 3.10+，只使用標準函式庫，不需要 GPU 或額外安裝套件。

範例（在 dfu_referral_starter 目錄執行）：
  python copy_images_by_result.py --predictions runs/b0_mtl_v1/test/predictions.csv --data-root data

預設輸出至 predictions.csv 旁的 classified_images 資料夾。
如需再次匯出，使用 --out 指定另一個資料夾；不覆寫既有結果。
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath


CATEGORIES = {
    (1, 1): "TP",
    (0, 0): "TN",
    (0, 1): "FP",
    (1, 0): "FN",
}
MEANINGS = {
    "TP": "標記需轉診，預測需轉診：正確找出",
    "TN": "標記不需轉診，預測不需轉診：正確排除",
    "FP": "標記不需轉診，預測需轉診：誤報",
    "FN": "標記需轉診，預測不需轉診：漏判",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_value(value, field, line):
    text = str(value).strip()
    if text in ("0", "0.0"):
        return 0
    if text in ("1", "1.0"):
        return 1
    raise ValueError(f"CSV 第 {line} 列的 {field} 必須是 0 或 1，目前為 {value!r}。")


def export_images(args):
    predictions = Path(args.predictions).expanduser().resolve(strict=True)
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    out = (Path(args.out).expanduser() if args.out else predictions.parent / "classified_images").absolute()
    if not data_root.is_dir():
        raise ValueError(f"--data-root 必須是資料夾：{data_root}")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"輸出位置已存在：{out}。請用 --out 指定新的資料夾。")

    predictions_hash = sha256(predictions)
    with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"image_path", "label", "prediction"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"CSV 缺少欄位：{', '.join(sorted(missing))}。請使用 evaluate 產生的 predictions.csv。")
        if len(fields) != len(set(fields)):
            raise ValueError("CSV 含有重複的欄位名稱。")
        if {"result_group", "copied_image_path"} & set(fields):
            raise ValueError("請使用原始 predictions.csv，不要使用本工具產生的 index.csv。")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV 沒有影像資料。")
    if sha256(predictions) != predictions_hash:
        raise ValueError("讀取時 CSV 發生變更，請待檔案寫入完成後再執行。")

    # 先確認全部影像、標籤與雜湊，再開始複製。
    plans, seen = [], set()
    for index, row in enumerate(rows, start=1):
        line = index + 1
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV 第 {line} 列的欄位數不一致。")
        label = binary_value(row["label"], "label", line)
        prediction = binary_value(row["prediction"], "prediction", line)
        raw_path = row["image_path"].strip().replace("\\", "/")
        relative = PurePosixPath(raw_path)
        if not raw_path or relative.is_absolute() or PureWindowsPath(raw_path).drive or ".." in relative.parts:
            raise ValueError(f"CSV 第 {line} 列的 image_path 必須是 data-root 下的相對路徑：{raw_path!r}")
        source = (data_root / Path(*relative.parts)).resolve(strict=True)
        if not source.is_relative_to(data_root) or not source.is_file():
            raise ValueError(f"無效影像路徑，或檔案位於 data-root 之外：{raw_path}")
        if source in seen:
            raise ValueError(f"CSV 重複列出同一張影像：{raw_path}")
        seen.add(source)
        source_hash = sha256(source)
        recorded_hash = row.get("file_sha256", "").strip().lower()
        if recorded_hash and recorded_hash != source_hash:
            raise ValueError(f"原始影像與預測紀錄的 SHA256 不符：{raw_path}")
        category = CATEGORIES[(label, prediction)]
        # 加上 CSV 列序號，避免不同子資料夾的同名照片互相覆蓋。
        copied_path = f"{category}/{index:04d}_{source.name}"
        plans.append((row, source, source_hash, category, copied_path))

    counts = Counter(plan[3] for plan in plans)
    summary = {
        "source_predictions": str(predictions),
        "source_predictions_sha256": predictions_hash,
        "data_root": str(data_root),
        "n_images": len(plans),
        "counts": {key: counts[key] for key in MEANINGS},
        "meanings": MEANINGS,
        "label_definition": "0=醫師標記不需轉診；1=醫師標記需轉診；未定義為緊急轉診",
        "decision_rule": "直接使用 CSV 的 prediction 欄位，不以 score 重算門檻",
        "copy_method": "複製原始檔案，不修改影像；每張複本驗證 SHA256",
        "output_directory": str(out),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # 在暫存目錄完成後才交付最終資料夾，避免混入未完成的匯出。
    with tempfile.TemporaryDirectory(prefix=".dfu_copy_", dir=out.parent) as temp:
        staging = Path(temp) / "classified_images"
        staging.mkdir()
        for category in MEANINGS:
            (staging / category).mkdir()
        with (staging / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields + ["result_group", "copied_image_path"])
            writer.writeheader()
            for row, source, source_hash, category, copied_path in plans:
                target = staging / copied_path
                shutil.copy2(source, target)
                if sha256(target) != source_hash:
                    raise ValueError(f"複製驗證失敗，原始檔可能在過程中被修改：{source}")
                writer.writerow({**row, "result_group": category, "copied_image_path": copied_path})
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"輸出位置已存在：{out}。請用 --out 指定新的資料夾。")
        staging.rename(out)

    print(f"完成：{out}")
    for category, meaning in MEANINGS.items():
        print(f"  {category}: {counts[category]:4d} 張  {meaning}")
    print(f"總計：{len(plans)} 張；原始照片保持不變。")
    print("index.csv：每張照片的預測紀錄與複製位置；summary.json：各類數量。")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="evaluate 產生的 predictions.csv")
    parser.add_argument("--data-root", default="data", help="原始影像根目錄，預設 data")
    parser.add_argument("--out", help="輸出資料夾，預設 CSV 旁的 classified_images；必須尚未存在")
    args = parser.parse_args()
    try:
        export_images(args)
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
