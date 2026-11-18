# DFU Foot ML：糖尿病足影像轉診分類

以 **EfficientNet-B0** 與 ImageNet 預訓練權重，學習從糖尿病足傷口照片預測人工標記的轉診需求：**0 = 不需轉診、1 = 需轉診**。

本專案比較「轉診分類」與「轉診分類＋傷口分割輔助任務」。兩者均為**監督式學習**，預測新影像時都只需要照片，不需要人工標籤或傷口遮罩。

> **版本範圍：這個 repository 目前是 WSL 的 V1 程式。** 本文以 V1 標籤及 `splits_dfu2` 說明，沒有包含另一台 Linux 電腦的 V2 `run_pipeline.py`。V2 的標籤、樣本數與結果不能直接套用到本頁。研究目標是「轉診需求」，未定義為「緊急轉診」；目前結果尚未確認新病人泛化能力。

## 1. 程式與資料是否齊全

| 檔案 | 用途 |
|---|---|
| [import_dfu2.py](import_dfu2.py) | 讀取 DFU2 標籤 Excel、配對影像、排除未知標籤及完全重複副本、建立固定切分 |
| [dfu.py](dfu.py) | 純影像分類：訓練、Test 評估與單張影像預測；也提供通用病人 CSV 切分功能 |
| [dfu_multitask.py](dfu_multitask.py) | 準備遮罩、多任務訓練、Validation 比較、Test 評估與影像預測；需與 `dfu.py` 放在同一層 |
| [app.py](app.py) | 純分類模型的本機 Streamlit 預測介面；這個 V1 介面不支援多任務 checkpoint |
| [review_errors.py](review_errors.py) | 根據已儲存的預測，逐張檢查 TP／TN／FP／FN，另存人工檢查紀錄 |
| [copy_images_by_result.py](copy_images_by_result.py) | 依已儲存的分類結果，將原始照片複製到四個資料夾 |
| [check_env.py](check_env.py) | 檢查 Python、套件與 CUDA 狀態 |
| [requirements.txt](requirements.txt) | 固定非 PyTorch 的直接依賴版本；PyTorch 依 GPU／CPU 另外安裝 |
| [tests/](tests/) | 資料匯入、切分與分類流程的 11 項單元測試 |
| [data/labels_example.csv](data/labels_example.csv) | 通用病人 CSV 的**虛構格式範例**；不含照片，也不是 DFU2 的實際病人對應表 |

原始照片、標籤 Excel、傷口遮罩、訓練權重及產生的研究結果需另備。本 repository 未附上這些檔案；它們缺席不代表少了 Python 程式。請將有權使用的研究資料存於本機，`.gitignore` 已排除常見資料與結果目錄。

本次補檔與驗證記錄見 [REPOSITORY_CHECK.md](REPOSITORY_CHECK.md)。

## 2. 兩種模型如何訓練

| 設定 | `image + referral_mark` | `image + mask + referral_mark` |
|---|---|---|
| 執行程式 | `dfu.py` | `dfu_multitask.py` |
| 模型輸入 | RGB 傷口照片 | RGB 傷口照片 |
| 監督答案 | 人工轉診標籤 | 人工轉診標籤＋傷口遮罩 |
| 結構 | EfficientNet-B0＋分類頭 | 共用 EfficientNet-B0＋分類頭＋分割分支 |
| Total loss | Referral BCE | Referral BCE + 0.5 × Segmentation loss |
| Segmentation loss | 不適用 | Pixel BCE + Soft Dice loss |
| 新照片預測時 | 只輸入照片 | 只輸入照片 |

分類損失採 `BCEWithLogitsLoss`。分割遮罩提供像素層級的參考答案，透過分割損失影響共用影像特徵；不是把遮罩當作額外影像通道餵給模型。`0.5` 是分割損失的權重，不代表兩個任務各占 50%。

兩種設定都從 ImageNet 預訓練權重開始，並各自訓練：

1. **Head：前 5 個 epochs。** 凍結特徵提取網路及其 BatchNorm 統計；純分類版只訓練分類頭，多任務版同時訓練分類頭與分割分支。
2. **Finetune：最多 30 個 epochs。** 從 Head 階段驗證分類 AP 最佳的權重解凍微調；Head learning rate 為 `1e-3`，Finetune 為 `1e-5`。
3. **模型選擇。** 保存 Validation 分類 AP 最高的模型；微調階段 AP 連續 5 個 epochs 未改善便提前停止。多任務模型的選擇依據也是分類 AP，並非分割 Dice。
4. **門檻選擇。** 預設以 Validation 的最大 F1 選擇分類門檻，再固定門檻評估 Test。預設並不保證 Sensitivity 達到 90%。

共同設定：224×224 等比例縮放與補邊、ImageNet normalization、訓練影像水平翻轉、AdamW、weight decay `1e-4`、seed `42`。下方指令統一使用 batch size `8`；`dfu.py` 本身的預設 batch size 是 `16`，所以指令明確覆寫。未啟用 `--class-weight`。

## 3. DFU2 V1 資料

本流程只選取 Excel 對應的 **DFU2** 影像；不混入 DFU1、DFU3 或 thermal_DFU。

| Split | 原始影像 | 排除 `?` | 排除已標記的重複副本 | 最終影像 | Label 0 | Label 1 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 810 | 90 | 20 | 700 | 220 | 480 |
| Validation | 200 | 25 | 1 | 174 | 61 | 113 |
| Test | 200 | 7 | 7 | 186 | 59 | 127 |
| Total | 1,210 | 122 | 28 | 1,060 | 340 | 720 |

以上是 V1 原始標籤的既有稽核結果，詳細來源與 SHA-256 見 [DATA_AUDIT_20261111.md](DATA_AUDIT_20261111.md)。更換標籤檔後，請以重新匯入的 `audit_report.json` 為準。

- `?`／空白標籤不當成 0；這些影像不進入訓練、驗證或測試清單，原始檔案仍保留。
- 保留來源既有 split，不重新隨機切分。完全相同像素的影像按 Test > Validation > Train 的順序保留一份；同組再按檔名排序。相同影像若有衝突的 0／1 標籤，匯入會停止。
- **沒有 patient_id。** 此處的 700／174／186 是影像數，不是病人數；完全重複影像去除不等於已確認病人互不重疊。
- 多任務流程只準備 Train／Validation 的遮罩。未確認有效的全黑遮罩不計分割損失及 Dice，但影像仍用於分類；分割損失也排除人工補邊。

## 4. 安裝：WSL Ubuntu／Linux

以下指令在 WSL／Linux 終端機執行。已在 WSL 提示字元中時，不需再輸入 `wsl`。

```bash
git clone https://github.com/lenkun25369-ui/DFU_foot_ML.git
cd DFU_foot_ML

conda create -n dfu_referral python=3.10 -y
conda activate dfu_referral

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python check_env.py
nvidia-smi
```

如果已建立 `dfu_referral` 環境，可直接啟用並檢查依賴，不必重建。安裝指令沿用專案使用的固定版本，並非最新版本宣稱。

僅使用 CPU 時，將上方 PyTorch 安裝指令替換為：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

後續訓練、評估及預測的 `--device cuda` 改成 `--device cpu`。首次正式訓練會下載 ImageNet 預訓練權重；`--random-init` 僅供軟體測試，不用於正式研究。

若 conda 提示字元與實際 Python 不一致，可檢查：

```bash
type -a python
python -c "import sys; print(sys.executable)"
```

## 5. 準備本機資料

在 repository 根目錄建立 `raw_data/`，自行放入以下兩個 V1 原檔：

```bash
mkdir -p raw_data
```

| 本機路徑 | 內容 |
|---|---|
| `raw_data/DFU2_train810_val200_test200_標籤.xlsx` | 含 `split`、`image_name`、`zip_path`、`label` 的標籤表 |
| `raw_data/drive-download-20261111T024845Z-1-001.zip` | 原始影像包；多任務流程還需要其中的 DFU2 傷口遮罩 |

若檔名不同，請替換後續指令中的路徑。程式可處理原 Excel 的 `ï»¿split` 欄名，並將 Excel 的 `val` 對應到影像包的 `validation`。

**資料中的兩種 label 不同：** Excel 的 `label` 是每張照片的轉診答案；ZIP 中 `DFU2/train/labels/`、`DFU2/validation/labels/` 的影像則是傷口遮罩，需與對應 `images/` 的影像同名。

```bash
python import_dfu2.py \
  --excel 'raw_data/DFU2_train810_val200_test200_標籤.xlsx' \
  --source-zip 'raw_data/drive-download-20261111T024845Z-1-001.zip' \
  --data-root data \
  --out splits_dfu2 \
  --label-definition '人工標記之轉診需求：0=不需轉診；1=需轉診；未定義為緊急轉診'
```

匯入後，照片位於 `data/DFU2/`；程式自動建立 `splits_dfu2/train.csv`、`val.csv`、`test.csv`、`manifest.json` 及稽核報告，不必手工建立 CSV。

`manifest.json` 記錄切分、標籤定義及檔案雜湊，用於確認訓練、評估使用同一版本的資料。不要手動改 CSV 後沿用舊 manifest／checkpoint。

純分類的匯入支援 `--source-dir` 取代 `--source-zip`；**此 V1 的 `prepare-masks` 仍需要原始 ZIP**。

## 6. 訓練純分類模型

```bash
python dfu.py train \
  --data-root data \
  --splits splits_dfu2 \
  --out runs/b0_dfu2_v1 \
  --device cuda \
  --batch-size 8 \
  --head-epochs 5 \
  --finetune-epochs 30
```

完成後保存 `runs/b0_dfu2_v1/model.pt`。訓練階段只載入 Train／Validation 影像；此時還沒有 Test 成績。

## 7. 準備遮罩並訓練多任務模型

先產生遮罩清單與 `masks_dfu2/manifest.json`：

```bash
python dfu_multitask.py prepare-masks \
  --source-zip 'raw_data/drive-download-20261111T024845Z-1-001.zip' \
  --data-root data \
  --splits splits_dfu2 \
  --out masks_dfu2
```

再訓練：

```bash
python dfu_multitask.py train \
  --data-root data \
  --splits splits_dfu2 \
  --masks masks_dfu2 \
  --out runs/b0_mtl_v1 \
  --device cuda \
  --batch-size 8 \
  --head-epochs 5 \
  --finetune-epochs 30 \
  --seg-weight 0.5
```

多任務模型獨立從 ImageNet 初始化，沒有接續 `b0_dfu2_v1/model.pt` 訓練。若出現 `masks_dfu2/manifest.json` 不存在，先完成上方 `prepare-masks`，不要自行建立空 JSON。

## 8. 先比較 Validation，再固定模型評估 Test

兩個模型都訓練完成後：

```bash
python dfu_multitask.py compare \
  --baseline runs/b0_dfu2_v1 \
  --candidate runs/b0_mtl_v1 \
  --splits splits_dfu2 \
  --specificity 0.80 \
  --out comparisons/b0_mtl_v1
```

**`compare` 只讀 Validation 結果。** `--specificity 0.80` 是研究比較時的最低目標；實際 Specificity 可能高於 0.80，也可能在兩模型間不同。此指令不會改動 checkpoint 的門檻；0.80 並非經驗證的臨床標準。

確定設定後，分別執行 Test 評估：

```bash
python dfu.py evaluate \
  --checkpoint runs/b0_dfu2_v1/model.pt \
  --data-root data \
  --splits splits_dfu2 \
  --device cuda \
  --batch-size 8

python dfu_multitask.py evaluate \
  --checkpoint runs/b0_mtl_v1/model.pt \
  --data-root data \
  --splits splits_dfu2 \
  --device cuda \
  --batch-size 8
```

多任務 Test 評估不使用真實傷口遮罩。`at_selected_threshold` 是 Validation 選定並儲存的門檻；`at_0_5` 是固定 0.5 的另一組報告，兩者不要混用。

既有研究已看過這批 Test；後續比較屬探索性分析。不要依 Test 成績反覆調整門檻後，仍把它稱為未使用過的最終測試集。

## 9. 結果存在哪裡？圖是 Validation 還是 Test？

下表的 `<run>` 分別為 `runs/b0_dfu2_v1` 或 `runs/b0_mtl_v1`。

| 路徑 | 內容／資料來源 |
|---|---|
| `<run>/model.pt` | 選定模型、前處理設定、分類門檻及資料切分雜湊 |
| `<run>/config.json` | 訓練參數及套件版本 |
| `<run>/history.csv` | 每個 epoch 的訓練與驗證紀錄 |
| `<run>/training_curves.png` | **Train／Validation** 訓練曲線，不含 Test 曲線 |
| `<run>/validation_metrics.json` | **Validation** 指標；參與模型／門檻選擇 |
| `<run>/validation_predictions.csv` | 每張 **Validation** 影像的分數、標籤與預測 |
| `comparisons/b0_mtl_v1/comparison.csv` | 兩個模型的 **Validation** 比較 |
| `<run>/test/metrics.json` | **Test** 指標 |
| `<run>/test/evaluation.png` | **Test** ROC、Precision–Recall curve、Confusion matrix |
| `<run>/test/predictions.csv` | 每張 **Test** 影像的分數、標籤與預測 |
| `<run>/test/false_negatives.csv` | **Test** 漏判影像清單 |

`test/evaluation.png` 上的 **Validation-selected threshold** 表示「門檻來自 Validation」；圖中的影像及成績仍來自 **Test**。

## 10. 依 TP／TN／FP／FN 複製照片

使用 Test 已儲存的 `prediction`，不重新推論或改門檻：

```bash
python copy_images_by_result.py \
  --predictions runs/b0_dfu2_v1/test/predictions.csv \
  --data-root data

python copy_images_by_result.py \
  --predictions runs/b0_mtl_v1/test/predictions.csv \
  --data-root data
```

兩個模型分別輸出到自己 `test/classified_images/` 下的 `TP/`、`TN/`、`FP/`、`FN/`，並附上 `index.csv` 與 `summary.json`。照片是複製，原始影像不變。

| 資料夾 | 人工標籤 | 模型預測 | 意義 |
|---|---|---|---|
| TP | 需轉診 | 需轉診 | 正確找出 |
| TN | 不需轉診 | 不需轉診 | 正確排除 |
| FP | 不需轉診 | 需轉診 | 誤報 |
| FN | 需轉診 | 不需轉診 | 漏判 |

逐張檢查與另存人工紀錄：

```bash
python -m streamlit run review_errors.py --server.address 127.0.0.1 -- \
  --predictions runs/b0_dfu2_v1/test/predictions.csv \
  --data-root data \
  --review-out reviews/b0_dfu2_v1_review.csv
```

要檢查多任務結果，將 `--predictions` 與 `--review-out` 內的 `b0_dfu2_v1` 改成 `b0_mtl_v1`。

## 11. 預測一張新照片

請將 `/path/to/new_foot.jpg` 換成實際照片路徑。

```bash
python dfu.py predict \
  --checkpoint runs/b0_dfu2_v1/model.pt \
  --image '/path/to/new_foot.jpg' \
  --device cuda

python dfu_multitask.py predict \
  --checkpoint runs/b0_mtl_v1/model.pt \
  --image '/path/to/new_foot.jpg' \
  --device cuda
```

輸出包含 `score`、`threshold`、`prediction`（0／1）。模型分數未經臨床風險校正，不宜直接解讀成「需要轉診的百分比」。

純分類模型也可使用本機網頁：

```bash
python -m streamlit run app.py --server.address 127.0.0.1 -- \
  --checkpoint runs/b0_dfu2_v1/model.pt
```

開啟 `http://localhost:8501`。此 V1 `app.py` 只支援純分類模型；多任務模型請使用上方 `dfu_multitask.py predict`。

## 12. 指標怎麼看

| Metric | 淺白意義 |
|---|---|
| Accuracy | 全部影像中，總共判對多少比例；容易受到兩類比例影響 |
| Sensitivity / Recall | 人工標記需轉診的影像中，模型成功找出多少比例 |
| False Negative Rate (FNR) | 人工標記需轉診的影像中，模型漏掉多少比例；`FN / (TP + FN) = 1 − Sensitivity` |
| Specificity | 人工標記不需轉診的影像中，模型正確排除多少比例 |
| Precision / PPV | 模型預測需轉診的影像中，有多少符合人工需轉診標籤 |
| NPV | 模型預測不需轉診的影像中，有多少符合人工不需轉診標籤 |
| F1 | Precision 與 Recall 的調和平均；不直接反映 TN |
| AUROC | 需轉診影像的分數通常能否排在不需轉診影像前面；不是某個門檻下的 Accuracy |
| Average Precision (AP) | 綜合不同 Recall 水準下的 Precision；受正類比例影響，並非梯形法 PR-AUC |
| Dice | 預測傷口區域與遮罩的重疊程度；不是轉診分類準確率 |

**Accuracy 變高仍可能伴隨更高的漏判率。** 尤其不需轉診影像較多時，正確判出這一大類即可提高 Accuracy；應同時查看 Sensitivity、FNR、Specificity 與混淆矩陣。不同標籤版本的類別比例及測試影像改變時，不能只比 Accuracy 就說模型改善。

## 13. 已有 V1 實驗紀錄

以下是研究者先前提供的 WSL 執行結果，非本次 repository 檢查重新訓練所得。兩者使用相同的 **186 張 Test 影像（Label 1：127；Label 0：59）**，各用自己的 Validation-selected threshold。原始 checkpoint 與逐張結果未隨 repository 發布。

| Metric | `image + referral_mark` | `image + mask + referral_mark` |
|---|---:|---:|
| Best epoch | 20 | 19 |
| Early stopping epoch | 25 | 24 |
| Threshold | 0.525720 | 0.450838 |
| Accuracy | 66.67% | 63.44% |
| Sensitivity | 62.20% | 56.69% |
| Specificity | 76.27% | 77.97% |
| AUROC | 0.747 | 0.763 |
| Average Precision | 0.872 | 0.877 |
| TN / FP / FN / TP | 45 / 14 / 48 / 79 | 46 / 13 / 55 / 72 |
| False Negative Rate | 48 / 127 = 37.80% | 55 / 127 = 43.31% |

本次加入分割監督後，AUROC／AP 小幅提高，但在各自選定的門檻下，漏判增加；尚未顯示能改善轉診漏判問題。此結果是單次探索性實驗，未確認病人獨立性，也不代表已完成臨床驗證。

## 14. 常見問題與重現限制

- **找不到 `requirements.txt`：** repository 已包含此檔案；請同步最新版本，並在 repository 根目錄執行安裝指令。
- **找不到 `runs/b0_mtl_v1/model.pt`：** 先完成多任務訓練，才能 `compare` 或 `evaluate`。只訓練純分類版不會產生多任務權重。
- **輸出資料夾已存在：** 程式會拒絕覆寫。重新實驗時指定新的 `--out`，並同步修改後續命令的 checkpoint／結果路徑；現有 V1 訓練沒有中斷續訓功能。
- **更換了標籤：** 重新建立 split、遮罩清單與訓練 run；不要直接覆蓋舊 CSV 或混用舊 checkpoint。
- **沒有 patient_id：** `splits_dfu2/patient_map_template.csv` 是空白待填表，不代表已取得病人 ID。需向資料來源取得真實研究用代碼；同一病人的不同照片應共用 ID。不能用檔名當病人 ID。
- **新病人驗證：** 若補齊病人對應後發現跨組，應重新規劃病人層級切分、訓練與驗證。目前不輸出未經驗證的病人群組信賴區間。
- **需要 Linux V2：** 本 repository 尚未整合 `run_pipeline.py`、`setup_env.sh`、`inspect_labels.py` 及 V2 修改後的模型／匯入程式；不能只補 `run_pipeline.py` 就視為完整 V2。

## 15. 軟體檢查與其他文件

```bash
python -m unittest discover -s tests -v
```

11 項單元測試使用合成資料，檢查資料配對、未知標籤、重複影像、病人切分、門檻與 checkpoint 等行為；通過測試不等於已證明模型的臨床效能。本次檢查範圍見 [REPOSITORY_CHECK.md](REPOSITORY_CHECK.md)。

- [README_zh_TW.md](README_zh_TW.md)：原始 V1 操作紀錄，以純分類與原本本機路徑為主。
- [GUIDE_patient_csv.md](GUIDE_patient_csv.md)：取得真實病人對應後的通用 CSV 流程；不是目前 DFU2 固定切分的必要步驟。
- [DATA_AUDIT_20261111.md](DATA_AUDIT_20261111.md)：V1 原始資料稽核。
- [VALIDATION.md](VALIDATION.md)：2026-11-11 程式交付時的軟體驗證紀錄；其「尚未完整訓練」描述指當時交付狀態。
