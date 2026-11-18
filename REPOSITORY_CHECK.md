# Repository 檔案檢查紀錄

檢查日期：2026-11-18。初始基準：`main` 的 commit `845c03263fb27aa4c76bad8c29be1edd9fedaa18`。檢查期間，使用者在 `0de396dba07e5a0171799468d6632610516b8f61` 補上 `requirements.txt`；已核對該次變更，以下補傳內容以此版本為基礎準備。

交付狀態：GitHub 連線的寫入要求遭 403 拒絕，本次新增檔案以補傳包交付，尚未由助理提交到 repository。

## 檢查結論

上傳內容是 **WSL V1 專案**。原有 13 個檔案中，7 個根目錄 Python 程式與 2 個測試程式已涵蓋目前的資料匯入、分類訓練、多任務訓練、評估、預測、影像分類匯出與錯誤檢視流程。

`dfu.py`、`import_dfu2.py`、`dfu_multitask.py` 及 `copy_images_by_result.py` 的文字內容與先前交付的 V1 對應檔一致（忽略檔尾空白）。檢查 Python imports 後，專案內部需要的 `dfu`、`import_dfu2` 模組均已存在。

此檢查以 GitHub 已上傳的檔案與先前交付程式為依據，未直接連線讀取使用者 WSL 硬碟；無法據此列出使用者本機後來另增的全部檔案。

## 本次補齊的檔案

| 檔案 | 原先狀態 | 本次處理 |
|---|---|---|
| `requirements.txt` | 初次檢查時缺少；使用者於檢查期間補上 | 已核對為原 V1 的 7 個固定非 PyTorch 依賴並保留；PyTorch 安裝來源由 README 說明 |
| `data/labels_example.csv` | 原病人 CSV 教學的範例未上傳 | 恢復虛構格式範例，明確說明不含影像與真實病人對應 |
| `README.md` | 沒有統整兩種模型的首頁文件 | 新增安裝、資料準備、兩種訓練、Validation／Test、輸出位置、錯誤檢查及既有 V1 結果 |
| `.gitignore` | 缺少 | 排除本機資料、模型權重、結果、環境與快取；保留虛構 CSV 範例 |
| `REPOSITORY_CHECK.md` | 缺少本次檢查紀錄 | 記錄檢查基準、缺檔處理、版本範圍及驗證限制 |

本次新增說明、虛構範例與 Git 忽略規則，保留使用者剛補上的依賴清單；訓練及推論程式維持上傳版本。

## 需在本機另備的研究檔案

| 內容 | 用途／取得方式 |
|---|---|
| DFU2 影像 ZIP、轉診標籤 Excel | 執行資料匯入與正式訓練的輸入 |
| ZIP 中 Train／Validation 的傷口遮罩 | 執行多任務訓練的監督答案 |
| `splits_dfu2/` | 執行 `import_dfu2.py` 自動產生，包含 CSV、manifest、稽核與排除清單 |
| `masks_dfu2/` | 執行 `dfu_multitask.py prepare-masks` 自動產生 |
| `runs/.../model.pt` | 訓練後產生；若要直接使用既有模型，需取得相符的 checkpoint |
| `runs/.../test/` | 執行 `evaluate` 產生 Test 指標、預測 CSV 與圖表 |
| 已填好的病人對應表 | 需由資料來源提供真實研究用 ID；空白 template 不是病人識別資料 |

上述研究輸入與產物不是缺少的 Python 模組，也不是一般程式碼補傳的必要項目。要在別台機器完整重現既有實驗，仍需另行取得對應版本的研究輸入、切分與權重。

## V1／V2 的範圍

這次檢查的 V1 不包含另一台 Linux 電腦專案的 `run_pipeline.py`、`setup_env.sh`、`inspect_labels.py`、`tests/test_v2.py`、`tests/smoke_workflow.py`，也不是 V2 修改後的 `dfu.py`、`dfu_multitask.py`、`import_dfu2.py`、`app.py`。

這些不是目前 V1 指令的必要依賴。若之後要納入 Linux V2，應以清楚的版本／目錄組織整套程式與文件，再驗證相容性；單獨補上 V2 wrapper 不構成完整升級。

## 本次驗證

- 以 Linux CPU 環境執行 `python -m unittest discover -s tests -v`：**11 項測試全部通過**。本次環境為 Python 3.12.14、torch 2.5.1+cpu、torchvision 0.20.1+cpu；7 個直接依賴與補回的 `requirements.txt` 版本一致。
- 以現有程式的 argument parser 核對 README 中的資料匯入、訓練、比較、評估與預測指令；核對影像匯出工具的參數。
- 檢查 README 的 repository 相對連結、Python 語法，以及 `.gitignore` 對研究資料／輸出與 CSV 範例的處理。
- 軟體測試使用合成資料。GPU 執行、使用者 WSL 本機環境及真實影像完整訓練不在本次驗證範圍。

README 的 V1 效能數字取自研究者先前提供的真實執行紀錄，並非上述單元測試產生。原始 [VALIDATION.md](VALIDATION.md) 保留為 2026-11-11 交付時的歷史軟體驗證紀錄。
