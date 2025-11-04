<!--
OpenSpec 提案：垃圾郵件分類 — Phase1 基線 (SVM + Streamlit)
作者：你 / 專案
日期：2025-11-04
狀態：草案
-->
# 0002：垃圾郵件分類 — Phase1 基線（SVM + Streamlit）

## 一句話摘要

以支援向量機（SVM，LinearSVC 或 LinearKernel）作為 Phase1 的基線分類器，使用公開 SMS spam 資料訓練模型，並以 Streamlit 建立互動式 demo：展示資料分布、訓練結果、輸入訊息即時預測是否為 spam，並允許使用者調整決策閾值（threshold）。

## 動機

- 快速建立一個可重現的 baseline，便於比較後續更複雜模型（例如 Logistic Regression 或 transformers）。
- SVM 在小到中等規模的文字分類任務表現穩定，是良好的起點。
- Streamlit demo 讓開發者與非技術利害關係人能直接互動並理解模型行為。

## 資料來源

使用下列公開 CSV（無標頭）作為基線訓練資料：

https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

註：請在廣泛發佈或商業使用前確認資料授權。

## Phase 1 範圍與細節

1) 資料擷取與前處理

   - 下載並快取 CSV 檔案。
   - 解析兩欄（label, message）；將標籤映射為二元（spam=1, ham=0）。
   - 基本文本淨化：小寫化、去除多餘空白、可選移除標點或停用詞。
   - 切分資料：train/validation/test（例如 70/15/15）並固定 random seed 以利可重現。

2) 特徵工程

   - 使用 TF-IDF 向量化（sklearn.feature_extraction.text.TfidfVectorizer），設定可配置參數如 max_features（預設 10000）、ngram_range=(1,2)。

3) 模型（SVM）

   - 主要：LinearSVC（scikit-learn）。
   - SVM 原生不支援 predict_proba；若需要機率分數以便 threshold 調整，可使用 `CalibratedClassifierCV` 對 SVM 進行校準（Platt scaling / isotonic）。
   - 若類別不平衡，使用 class_weight='balanced' 或重採樣策略。

4) 評估指標

   - 基礎：accuracy、precision、recall、F1。
   - 排序/閾值相關：ROC AUC、PR 曲線。
   - 額外顯示：confusion matrix、top tokens（係數或權重較高的特徵）

5) Streamlit UI 要求

   - 資料總覽：類別比例圖（bar chart）、範例訊息、訊息長度分佈。
   - 訓練結果：指標表格、混淆矩陣熱圖、ROC / PR 曲線、模型重要特徵（前 N token）。
   - 預測介面：文字輸入框（textarea）給使用者輸入訊息，按下 Predict 後顯示：
     - 預測機率（若使用校準器）或 decision score（若未校準），
     - 判定標籤（spam/ham）依據目前閾值。
   - 閾值滑桿（0.0–1.0）：即時更新在驗證集上的 precision/recall 與預測標籤。
   - 允許下載模型與向量器（pickle）與 metrics report。

6) 輸出與可重現性

   - 儲存 artefacts：`vectorizer.pkl`, `model.pkl`, `calibrator.pkl`（若有）, `metrics.json`。
   - 使用 deterministic seed 並提供 `requirements.txt`。

## 實作與測試（驗收標準）

- 管線可以下載資料並成功產生模型與向量器檔案。
- Streamlit demo 正常啟動並呈現資料分布、訓練指標、混淆矩陣與 ROC/PR 曲線。
- 預測介面能回傳 0–1 的機率（若校準）或得分，並依滑桿閾值正確決策。
- 單元測試：前處理輸出型別、vectorizer transform shape、model predict / decision_function 以及校準器輸出形狀、閾值化邏輯。

## 風險與注意事項

- SVM 若不校準，無法直接給出機率，會影響閾值化解釋；建議使用 `CalibratedClassifierCV` 作為一步驟以取得機率輸出。
- 若資料中包含個資，請勿在公開 demo 中使用真實敏感資料。

## 實作步驟（具體）

1. 新增 `requirements.txt`：
   - streamlit
   - scikit-learn
   - pandas
   - matplotlib
   - seaborn
   - joblib

2. 建立 `src/spam_classifier/` 模組：
   - `data.py`：下載、快取、解析 CSV
   - `preprocess.py`：清理文字、切分資料、建立並儲存 TF-IDF
   - `train.py`：訓練 LinearSVC、（選擇性）使用 CalibratedClassifierCV 校準、輸出 metrics
   - `predict.py`：載入模型與向量器、依閾值回傳標籤

3. 建立 `apps/streamlit_app.py`：顯示資料與模型指標，提供預測介面與閾值滑桿。

4. 新增 `tests/test_pipeline.py`：測試前處理、向量化、預測與閾值邏輯。

5. 撰寫 `README.md`：說明如何安裝與啟動（local streamlit run apps/streamlit_app.py）。

## 替代方案

- 直接使用 Logistic Regression（優點：原生支援 predict_proba，可直接 threshold）。
- 使用外部服務或預訓練模型（transformer）— 可提升效能但成本與複雜度顯著提高。

## 開放問題

- 是否要在 Phase1 直接對 SVM 做機率校準（建議：是）？
- UI 是否需要額外的視覺化（例如 word-cloud）或匯出為交互式報表？

## 時程估計

- 提案：0 天（此文件）
- 管線與測試：1 天（基礎版本）
- Streamlit demo：1 天
-（選項）容器化與 CI：+1 天

## 我可以幫你做的下一步

- 我可以立即 scaffold 程式檔案並實作訓練流程（含校準），然後在本機嘗試跑一次訓練並回報結果。
- 如果你要我開始實作，請回覆「開始訓練（SVM）」或「先 scaffold 檔案」。

