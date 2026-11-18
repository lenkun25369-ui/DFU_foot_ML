# 通用 CSV／病人分組流程

這是補齊真實 patient_id 後可採用的通用流程。你目前提供的 DFU2 Excel 請先依 README_zh_TW.md 操作。

這份專案從資料整理、病人分組、監督式訓練、保留測試集評估，一直到單張新影像推論與本機展示頁面。
輸入模型的是傷口照片；醫師標籤只用來計算訓練誤差與評估，不是額外輸入特徵。

假設環境：Windows + WSL2 Ubuntu + NVIDIA GPU。建立獨立 Python 環境，避免更動原有研究環境。
本專案不包含病人資料，也不包含已訓練的「轉診模型」。首次正式訓練會下載 ImageNet 預訓練權重。
所有步驟都在你自己的電腦執行。這是一個研究起始實作；實際病人資料上的成效尚待驗證。

## 0. 先準備資料資訊

- 每張照片的檔名、所屬病人 ID、當次醫師標籤。
- `1` = 醫師標記需要轉診；`0` = 醫師標記不需要轉診。
- 先確認指的是一般轉診、緊急轉診或其他明確的升級照護；不要混在同一個標籤。
- 同一病人的多張照片、多次回診，必須保有同一個 `patient_id`。
- 同一病人不同回診可以有不同標籤，需對應當次照片與評估。多張照片共享同一次決策時，記錄相同 `visit_id`。
- 病人 ID 只用於切分和評估，不會餵入模型；請使用研究用代碼。
- 1,000 張照片不等於 1,000 位獨立病人。先確認獨立病人數及兩類比例。
- 若沒有病人 ID，先找回可靠的對應關係；不要把每張照片的檔名當成不同病人。

## 1. 打開 WSL，確認 GPU

在 Windows PowerShell 輸入：

```powershell
wsl -l -v
wsl -d Ubuntu-22.04
```

如果第一行列出的 Ubuntu 名稱不同，第二行改用實際名稱。後面的指令都在 Ubuntu 執行。

```bash
nvidia-smi
conda --version
```

`nvidia-smi` 應列出 NVIDIA 顯示卡。若失敗，先處理 Windows/WSL GPU 支援；不要為此在 WSL 內另外安裝 Linux 顯示卡驅動。
若 conda 尚未初始化，但先前裝在 `~/anaconda3`，可先執行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda --version
```

若沒有 conda，可以使用第 3 步下方的 venv 替代流程。

## 2. 解壓縮專案

例如先把下載的壓縮檔存成 Windows 的 `D:\dfu_referral_starter.zip`。在 WSL 執行：

```bash
mkdir -p ~/projects
python3 -m zipfile -e /mnt/d/dfu_referral_starter.zip ~/projects
cd ~/projects/dfu_referral_starter
```

`D:\` 在 WSL 對應 `/mnt/d/`。如果存在同名專案，請先改用另一個目錄，避免解壓覆寫舊版本。

## 3. 建立獨立環境並安裝套件

優先使用已有的 conda：

```bash
conda create -n dfu_referral python=3.10 -y
conda activate dfu_referral
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python check_env.py
```

這是固定且相容的 PyTorch/torchvision 版本組合，並非宣稱為最新版本。之後升級時應一起升級並重跑測試。
CUDA 12.1 wheel 需要相容的 NVIDIA 驅動；不要根據舊專案的 PyTorch CUDA 版本判斷是否能安裝。
若 GPU 初始化失敗，保留錯誤訊息，先檢查 `nvidia-smi` 與 Windows 驅動。

預期會看到：

```text
CUDA available: True
GPU: NVIDIA ...
EfficientNet-B0 forward check: PASS (no pretrained download needed)
```

CPU 也可以跑軟體測試與推論，但完整訓練通常較慢。
如果確定只使用 CPU，於新的環境將上述 PyTorch 安裝行改成：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

沒有 conda 時，在 Ubuntu 22.04 使用以下替代方式；不必和 conda 流程一起做：

```bash
sudo apt update
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python check_env.py
```

日後重新開終端機，回到專案並啟用環境即可，不必重裝：

```bash
cd ~/projects/dfu_referral_starter
conda activate dfu_referral
```

若用 venv，最後一行改成 `source .venv/bin/activate`。

## 4. 放入照片並建立 labels.csv

建立照片目錄：

```bash
mkdir -p data/images
explorer.exe .
```

在開啟的檔案總管中，把照片複製到專案的 `data/images/`。
或在 WSL 用 `cp -a` 複製你的原始資料夾內容。先保留原始照片備份。

用 Excel 建立表格，另存為 **CSV UTF-8**，放到 `data/labels.csv`。
`data/labels_example.csv` 只有格式範例；其中檔名不存在，不能直接拿它訓練。

| image_path | patient_id | visit_id | label |
| --- | --- | --- | --- |
| images/P001_V001_01.jpg | P001 | V001 | 1 |
| images/P001_V001_02.jpg | P001 | V001 | 1 |
| images/P002_V001_01.jpg | P002 | V001 | 0 |

- 每列是一張照片，第一列一定是上述英文欄位名稱。
- `image_path` 以 `data/` 為起點，不要再寫一遍 `data/`，也不要填 Windows 的 `D:\...` 路徑。
- `patient_id` 不得空白；`visit_id` 可選，但有多次回診時建議保留。
- `label` 填數字 0 或 1，不填中文。
- JPG/JPEG/PNG 可用，程式會處理 EXIF 方向並轉 RGB。
- 照片若有浮水印、醫師註記、拍攝位置差異，需檢查是否洩露標籤；模型可能學到這些線索。

## 5. 檢查資料與建立固定切分

```bash
python dfu.py prepare --csv data/labels.csv --data-root data --out splits
```

若醫師標籤定義已確認，可明確寫入模型的研究紀錄。例如：

```bash
python dfu.py prepare --csv data/labels.csv --data-root data --out splits \
  --label-definition '醫師依當次完整評估標記一般轉診需求：0=不需；1=需'
```

上面兩個 prepare 指令擇一執行即可。若 `splits` 已存在，程式會拒絕覆寫。

此步驟會：

1. 確認照片存在、可以讀取、標籤只有 0/1、病人 ID 沒有空白。
2. 阻擋重複路徑及完全相同像素的影像，即使它們有不同檔名或壓縮格式。
3. 以病人為單位，建立約 70% 訓練、15% 驗證、15% 測試，盡量平衡類別。
4. 記錄每組照片數、病人數、類別分布和影像雜湊值，阻擋事後悄悄變更資料。

輸出是 `splits/train.csv`、`splits/val.csv`、`splits/test.csv` 與 `splits/manifest.json`。
比例依實際病人群組而定，不保證剛好 700/150/150 張；切分選擇不使用模型分數。
少數類病人過少、找不到三組各含兩類時會停止。請調整研究設計，不要拆散同一病人的照片。
完全重複檢查不等於近似重複檢查；旋轉、重新裁切或壓縮後的近似影像仍要人工稽核。

## 6. 訓練第一版

```bash
python dfu.py train --data-root data --splits splits --out runs/b0_v1 --device cuda
```

只有 CPU 時改用 `--device cpu`，也可省略 `--device` 讓程式自動選擇。
首次正式訓練會自 PyTorch 官方下載約 20 MB 的 ImageNet 權重；這是唯一必需的模型下載。

預設設定如下，都是研究起始值，並非已證明最適合你的資料：

| 項目 | 設定 |
| --- | --- |
| 模型 | EfficientNet-B0 / ImageNet1K 預訓練 |
| 輸入 | 224×224，等比例補邊，保留整張照片；使用 ImageNet normalization |
| 分類頭 | 1 個 logit；推論使用 sigmoid |
| 損失函數 | BCEWithLogitsLoss |
| 優化器 | AdamW；weight decay 0.0001 |
| Batch size | 16 |
| 第一階段 | 固定骨幹 5 epochs，只訓練分類頭；LR 0.001 |
| 第二階段 | 從第一階段最佳模型解凍微調，最多 30 epochs；LR 0.00001 |
| Early stopping | 微調階段驗證 AP 連續 5 epochs 未改善時停止 |
| 保存模型 | 兩階段中驗證 AP 最高的權重 |
| 增強 | 只有訓練集隨機水平翻轉；未使用強烈顏色改變或隨機裁切 |

固定骨幹時，也會固定 BatchNorm 的統計量。模型不讀取病人 ID 或 visit ID 作為特徵。
本專案採全圖補邊，與官方 ImageNet 範例的中心裁切不同；此選擇是為保留傷口周邊資訊，之後需驗證。

如果兩類非常不均衡，可以在另一個實驗加入 `--class-weight`；它只根據訓練集計算正類權重：

```bash
python dfu.py train --out runs/b0_weighted --device cuda --class-weight
```

不要同時再自行加過採樣。使用 class weight 時訓練 BCE 有加權、驗證 BCE 未加權，曲線的絕對大小不可直接比較。
若 GPU 記憶體不足，在新的實驗改用 `--batch-size 8` 或 4。
若中途停止，這個起始版不提供精確續訓；最佳 checkpoint 仍可讀取，但完成的門檻與驗證輸出才算整輪完成。

完成後的重要檔案：

| 檔案 | 內容 |
| --- | --- |
| runs/b0_v1/model.pt | 最佳權重、前處理大小、標籤定義、門檻、切分識別 |
| runs/b0_v1/config.json | 參數與套件版本 |
| runs/b0_v1/history.csv | 每個 epoch 的 loss、驗證 AUROC/AP |
| runs/b0_v1/training_curves.png | 訓練曲線 |
| runs/b0_v1/validation_metrics.json | 驗證集指標 |
| runs/b0_v1/validation_predictions.csv | 每張驗證影像的預測 |

訓練期間不會載入測試影像計算表現。多個模型/超參數應先在同一套訓練/驗證資料比較，再挑最終候選。

### 分類門檻怎麼選？

預設使用驗證集最大 F1 的門檻，並保留 0.5 門檻的結果作比較。這只是統計基準，不是臨床標準。
選模型與選門檻都用了驗證集，因此驗證成績有選擇偏差；最終成績以保留測試集為準。

若和醫師討論後，要研究「驗證敏感度至少 0.90 下，提高特異度」，可在新實驗使用：

```bash
python dfu.py train --out runs/b0_sensitivity --device cuda \
  --threshold-policy sensitivity --target-sensitivity 0.90
```

0.90 是可調的探索目標，不是安全門檻，也不保證在新病人維持 90%。
固定門檻後，不可以看測試答案再調整門檻。

## 7. 最後才評估保留測試集

選定最終模型/設定後執行：

```bash
python dfu.py evaluate --checkpoint runs/b0_v1/model.pt --data-root data --splits splits
```

輸出到 `runs/b0_v1/test/`：

- `metrics.json`：敏感度、特異度、precision、NPV、F1、AUROC、average precision，以及 95% 區間。
- `predictions.csv`：每張測試照片的預測，供逐例檢查。
- `false_negatives.csv`：醫師標記需轉診但被模型漏判的照片清單。
- `evaluation.png`：ROC、precision-recall 曲線與混淆矩陣。

報告的指標以「影像」為單位，信賴區間以整位病人為群組重抽樣，避免把同一病人的影像全當獨立樣本。
病人多張照片會對影像指標有較大權重；若實際任務是每次就診一個決策，需要另訂多張照片的彙整規則並驗證。
AP 指 average precision，並非梯形法計算的 PR-AUC。少量病人/正例會讓區間不穩定；請一起報告樣本數。
預設 bootstrap 500 次，供初步估計；正式研究可增加次數，並加入開發集內的病人分組交叉驗證或外部驗證。
測試集不是用來反覆選模型的。如果已看測試結果後再修改模型，那批資料就不再是未碰過的最終測試集。

## 8. 預測一張新照片

先把新照片存到例如 `data/new_foot.jpg`，它不需要 label：

```bash
python dfu.py predict --checkpoint runs/b0_v1/model.pt --image data/new_foot.jpg
```

輸出 JSON 包含：

- `score`：0 到 1 的模型分數，尚未校正，不能解讀為真正臨床風險百分比。
- `threshold`：訓練完成時由驗證集選定的門檻。
- `prediction`：0 或 1，對應你的醫師標籤定義。

本專案的推論會自動使用與訓練相同的方向校正、RGB、補邊與標準化。
請使用自己訓練產生的模型檔。

## 9. 啟動本機展示頁面

```bash
python -m streamlit run app.py -- --checkpoint runs/b0_v1/model.pt
```

在 Windows 瀏覽器開啟 `http://localhost:8501`。
選擇照片 → 點「執行模型預測」→ 查看分數、門檻與分類結果。
頁面使用 CPU 推論，以便和 GPU 訓練分開。上傳的照片由本機程式處理，不會由本程式另存到硬碟。
此設定只綁定本機 `127.0.0.1`，沒有部署到公網，也沒有帳號/院內系統整合。
終端機按 `Ctrl+C` 關閉。第一次展示建議使用保留的範例，並同時說明測試集的漏判與誤報結果。

## 10. 專案檔案與軟體檢查

| 檔案 | 用途 |
| --- | --- |
| check_env.py | 檢查套件、GPU 與 EfficientNet forward |
| dfu.py | prepare / train / evaluate / predict 四個步驟 |
| app.py | 本機預測展示頁面 |
| requirements.txt | 固定依賴版本；PyTorch 依 CPU/CUDA 另裝 |
| data/labels_example.csv | CSV 格式範例，非真實資料 |
| tests/test_pipeline.py | 分組、重複資料、門檻、模型存取等測試 |
| VALIDATION.md | 此版本已完成與尚未完成的驗證 |

可重跑軟體測試：

```bash
python -m unittest discover -s tests -v
```

`--random-init` 僅供離線測試程式，不應用於本研究的預訓練基準；正常指令不要加此參數。
套件執行成功與臨床有效是不同事項；目前驗證範圍請見 VALIDATION.md。

## 參考來源

- EfficientNet-B0 官方程式：https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
- 預訓練權重與官方前處理：https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.efficientnet_b0.html
- PyTorch 2.5.1 / torchvision 0.20.1 安裝配對：https://pytorch.org/get-started/previous-versions/
- NVIDIA 的 WSL 驅動安裝說明：https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- 遷移學習教學：https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- 病人分組切分使用的 GroupShuffleSplit：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- 門檻選擇：https://scikit-learn.org/stable/modules/classification_threshold.html
- Streamlit 本機檔案選擇元件：https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader

這份 pipeline 是依你的任務新寫的研究起始程式，不是已驗證的糖尿病足轉診臨床產品，也不是某篇 DFU 論文的原始實作。
