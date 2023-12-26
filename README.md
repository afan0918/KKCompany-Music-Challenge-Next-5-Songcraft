# KKCompany-Music-Challenge-Next-5-Songcraft

此競賽位於 https://www.kaggle.com/competitions/datagame-2023

我們的方案排名：1/172

最終設計了一個不會特別推薦熱門歌曲和運算速度自認為還算快的模型，希望能有些幫助。

## Contributors
|組員|貢獻內容|github|
|-|-|-|
|陳繹帆| ngrams+JMLM 解決方案研究 | [afan0918](https://github.com/afan0918) |
|陳品絜| MLE+pyserini 解決方案研究 | [pj-99](https://github.com/pj-99)|
|徐韶汶| EDA分析 | [jasonshen-python](https://github.com/jasonshen-python) |
|沈欣柏| 深度學習研究 | [AngelaHsu02](https://github.com/AngelaHsu02) |

## 運行環境

* ubuntu 22.04
* python 3.10.12

### 前置動作
將 datagame-2023 資料夾放置在本專案下，具體運行請參閱 final 資料夾內的 README.md


## Overview
歡迎參加我們的音樂推薦競賽！這個競賽旨在挑戰您的機器學習和數據分析技能，以改進音樂串流體驗。您的任務是預測用戶在同一個聆聽 session 內聆聽一定數量歌曲後，接下來可能會聆聽哪些歌曲。這將有助於打造更個性化的音樂推薦服務，提高用戶滿意度。我們特別強調參賽者需要謹慎避免過度集中於熱門音樂，以確保推薦結果更多元化，滿足不同用戶的需求。期待參賽者的創新方法，以改進音樂串流體驗，提高用戶滿意度。

Welcome to our Music Recommendation Competition! This competition is designed to challenge your machine learning and data analysis skills to enhance the music streaming experience. Your task is to predict what songs users might listen to after hearing a certain number of songs in the same listening session. This will contribute to creating a more personalized music recommendation service, improving user satisfaction. We particularly emphasize that participants need to avoid an excessive focus on popular music to ensure diverse recommendation results. We look forward to innovative approaches from participants to enhance the music streaming experience and increase user satisfaction.

## Description
在這場競賽中，您將獲得用戶的歌曲播放記錄和歌曲相關資訊的數據集。您需要建立一個預測模型，該模型基於用戶在同一個聆聽 session 內聆聽的前 N (=20) 首歌曲，預測接下來會聆聽哪 K (=5) 首歌曲。我們將使用 DCG（Discounted Cumulative Gain）和預測歌曲的覆蓋率（Coverage）來評估模型的性能。請注意，避免過度集中於熱門歌曲是本競賽的一個關鍵要求。參賽者可以使用任何合法外部資源和工具，以開發創新的解決方案，以改進音樂串流體驗，提高用戶滿意度。

In this competition, you will receive a dataset containing user song playback records and song-related information. Your task is to build a predictive model that, based on the first N=20 songs played within the same listening session, forecasts the subsequent listening choices. We will evaluate the model's performance using DCG (Discounted Cumulative Gain) and coverage. Please note that avoiding an excessive focus on popular songs is a key requirement of this competition. Participants are allowed to use any legitimate external resources and tools to develop innovative solutions to enhance the music streaming experience and increase user satisfaction. We look forward to seeing your submissions in this competition.

