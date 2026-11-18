# 你的 DFU2 資料：EfficientNet-B0 轉診分類

更新日期：2026-11-11。這版可直接讀取你提供的 Excel 與影像 ZIP，不必手動改成 CSV，也不必重新標註傷口位置。

模型採監督式學習：傷口照片是輸入、Excel 的 0/1 是訓練答案。採用 ImageNet 預訓練 EfficientNet-B0，訓練後可以輸入一張新照片預測標籤。

目前按先前研究方向設定 **0 = 不需轉診，1 = 需轉診**。Excel 本身沒有文字版標籤定義，請向標註者核對數字方向與轉診的時間／急迫性；程式不會自行把一般轉診改稱緊急轉診。若數字方向相反，需先在另存的 Excel 修正標籤，再重新匯入與訓練。

## 已核對的資料

| 原始分組 | 原始影像 | `?` 排除 | 已知標籤的重複副本排除 | 最終影像 | 最終 0／1 |
|---|---:|---:|---:|---:|---:|
| train | 810 | 90 | 20 | 700 | 220／480 |
| val | 200 | 25 | 1 | 174 | 61／113 |
| test | 200 | 7 | 7 | 186 | 59／127 |
| 合計 | 1,210 | 122 | 28 | 1,060 | 340／720 |

- 1,210 張都能與 ZIP 配對並讀取，原圖均為 512×512。
- Excel 欄位是 `split / image_name / zip_path / label`。程式處理 `ï»¿split` 的編碼問題，並用完整 `DFU2/分組/images/檔名` 配對；不依賴亂碼的外層資料夾名稱。
- Excel 的 `val` 對應 ZIP 內的 `validation`。
- ZIP 另有 DFU1、DFU3、熱影像及 `labels` 遮罩；本次只用 Excel 對應的 DFU2 `images`。
- 有 29 組完全相同像素，其中 7 組跨原始分組。去重時保留 test 優先，其次 val、train；同組則保留檔名字典序最前的一張。只排除副本，不把照片搬到另一組；原始 Excel 和 ZIP 不改寫。
- 同圖若有相互衝突的 0/1，匯入會停止，不自行決定誰正確。這次提供的資料沒有此衝突。
- `?` 不等於 0，也不透過遮罩或模型補答案；另外輸出待確認清單。評估適用於留下的已標記資料，不能直接推廣到未知標籤個案。

**目前沒有 patient_id。** 本版保留既有影像分組供探索；不捏造病人 ID、不聲稱病人互不重疊，也不輸出假病人群組的信賴區間。相近照片與同一病人跨組仍需另外核對。

## 1. 下載並放置檔案

從你分享的 Google Drive 下載兩個原檔，放進 Windows 的 `D:\dfu_data\`：

- `DFU2_train810_val200_test200_標籤.xlsx`
- `drive-download-20261111T024845Z-1-001.zip`

將本程式壓縮檔存到 `D:\dfu_referral_starter.zip`。在 WSL Ubuntu 執行：

```bash
mkdir -p ~/projects/dfu_20261111
python3 -m zipfile -e /mnt/d/dfu_referral_starter.zip ~/projects/dfu_20261111
cd ~/projects/dfu_20261111/dfu_referral_starter
```

若這個目錄已存在，換一個新目錄以免覆寫先前程式。後續指令都在這個專案目錄執行。

## 2. 環境

如果已經依上一版建立 `dfu_referral` 環境：

```bash
conda activate dfu_referral
python -m pip install -r requirements.txt
python check_env.py
```

本版新增 `openpyxl` 讀 Excel；其他固定依賴與上一版一致。

若還沒有環境，執行下列整段（與上一段擇一）：

```bash
conda create -n dfu_referral python=3.10 -y
conda activate dfu_referral
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python check_env.py
```

這是沿用專案已測試的版本組合，不是最新版本宣稱。WSL/CUDA 需相容的 NVIDIA 驅動；用 `nvidia-smi` 確認。若只有 CPU，PyTorch 安裝來源改為 `https://download.pytorch.org/whl/cpu`，下面訓練指定 `--device cpu`。更完整的 WSL、conda／venv 說明見 `GUIDE_patient_csv.md` 第 1–3 步。

## 3. 直接匯入你的 Excel 與 ZIP

```bash
python import_dfu2.py \
  --excel '/mnt/d/dfu_data/DFU2_train810_val200_test200_標籤.xlsx' \
  --source-zip '/mnt/d/dfu_data/drive-download-20261111T024845Z-1-001.zip' \
  --data-root data \
  --out splits_dfu2
```

不用先解壓縮 531 MB 的資料包。匯入只複製對應的 1,210 張 DFU2 照片到 `data/DFU2/`，原始照片與來源檔保留；需要約 306 MB 額外空間。未知標籤照片保留在磁碟上，方便之後人工標註，但不進入訓練／驗證／測試清單。

如果你已自行解壓縮資料，將上面的 `--source-zip ...` 改為 `--source-dir '/你的解壓縮目錄'`，其餘相同。目錄可以是含 DFU2 的上一層，或 DFU2 本身。

成功後應顯示 **train 700、val 174、test 186**。來源資料有變更時數目可能不同，請以新產生的稽核報告為準。

| 輸出至 splits_dfu2/ | 用途 |
|---|---|
| train.csv、val.csv、test.csv | 已去除未知答案與重複副本的固定清單 |
| manifest.json | 檔案雜湊、標籤定義、分組與病人驗證狀態 |
| audit_report.json | 匯入前後筆數及排除規則 |
| excluded_unknown.csv | 122 張待標註影像 |
| excluded_duplicates.csv | 28 個已標記但重複的副本，以及保留哪張 |
| duplicate_groups.json | 全部 29 組完全重複影像的明細 |
| patient_map_template.csv | 日後填入真實研究用病人 ID 的空白對應表 |

`--out` 目錄若已存在會拒絕覆寫。補標籤或修正資料後請改成例如 `--out splits_dfu2_v2`，並建立新的訓練 run；不能只改舊 CSV 卻沿用舊模型和測試結果。

## 4. 訓練 EfficientNet-B0

```bash
python dfu.py train \
  --data-root data \
  --splits splits_dfu2 \
  --out runs/b0_dfu2_v1 \
  --device cuda
```

這裡的 train 是 **700 張已標記且不完全重複的影像**，不是原始 810 張。未使用病人 ID、Excel 路徑、檔名或遮罩當模型特徵。

預設為 224×224 全圖等比例補邊、ImageNet normalization、batch size 16。先固定骨幹訓練分類頭 5 epochs，再以較小學習率微調最多 30 epochs；微調階段驗證 AP 連續 5 epochs 未改善便停止。只在訓練集做水平翻轉。

模型與分類門檻只使用 train/val 選擇，預設門檻取驗證集最大 F1。首次正式訓練下載 ImageNet 預訓練權重；不需傷口位置標註。若顯示記憶體不足，改用新的 `--out` 並加入 `--batch-size 8`。

產物包含 `model.pt`、`history.csv`、`training_curves.png`、`validation_metrics.json`、`validation_predictions.csv`。`model.pt` 同時保存權重、門檻、前處理與病人分組未驗證的狀態。

**沒有在此交付包內附上已訓練的轉診模型。** 已完成資料匯入與軟體測試，完整訓練在你的環境執行。不要在正式研究指令加 `--random-init`。

## 5. 模型設定選定後，才評估 test

```bash
python dfu.py evaluate \
  --checkpoint runs/b0_dfu2_v1/model.pt \
  --data-root data \
  --splits splits_dfu2
```

輸出到 `runs/b0_dfu2_v1/test/`，包含敏感度、特異度、AUROC、average precision、混淆矩陣、每張預測與漏判清單。這是 **186 張影像的探索性評估**；沒有 patient_id 時，病人數與病人群組 bootstrap 區間均為 `null`。

AP 是 average precision，並非梯形 PR-AUC。測試集不可拿來反覆選模型、改門檻；若已依測試結果調整模型，需重新規劃最終獨立驗證。

## 6. 輸入新照片

```bash
python dfu.py predict \
  --checkpoint runs/b0_dfu2_v1/model.pt \
  --image '/你的路徑/new_foot.jpg'
```

新照片不用 label。輸出 `score`、`threshold`、`prediction`（0/1）和標籤定義。分數尚未校正，不能直接當成臨床風險百分比。

要用網頁選照片：

```bash
python -m streamlit run app.py -- --checkpoint runs/b0_dfu2_v1/model.pt
```

在自己電腦開啟 `http://localhost:8501`。這是本機研究展示介面。

## 7. 之後補上 patient_id

`patient_map_template.csv` 可填入真實研究用代碼；同一病人的不同照片必須使用同一 ID，不可用照片流水號假裝病人 ID。

```bash
python import_dfu2.py \
  --excel '/mnt/d/dfu_data/DFU2_train810_val200_test200_標籤.xlsx' \
  --source-zip '/mnt/d/dfu_data/drive-download-20261111T024845Z-1-001.zip' \
  --patient-map '已填好的病人對應表.csv' \
  --data-root data \
  --out splits_dfu2_with_patients
```

所有 0/1 影像都必須完成病人對應。如果對應表顯示原始不同組其實有同一病人，程式會停止；需重新設計病人分組，不能直接沿用目前的測試集。通用病人分組模式見 `GUIDE_patient_csv.md`，重新分組前也需完成重複影像處理。

## 檔案與驗證

- `import_dfu2.py`：本次新增，直接讀你的 Excel／ZIP、配對、排除未知標籤及重複影像。
- `dfu.py`：訓練、評估、推論；保留舊的通用 `prepare` 病人分組功能。
- `app.py`：本機展示，同時顯示病人分組尚未驗證的限制。
- `DATA_AUDIT_20261111.md`：這批資料的稽核結果。
- `VALIDATION.md`：此次實際完成的軟體測試與驗證範圍。

```bash
python -m unittest discover -s tests -v
```

GitHub 模型實作：https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

這份專案使用公開的 EfficientNet-B0 架構，分類頭需用你的標籤訓練；不是現成且經臨床驗證的轉診系統。
