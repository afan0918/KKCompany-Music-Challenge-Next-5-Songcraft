# 如何運行

## pyserini(執行時間約 35 分鐘)

* 建立一個新的虛擬環境，python 版本盡量在 3.8 ~ 3.10之間。
* 安裝 jupyter
* 讓終端機中的路徑進到本資料夾
```shell
cd final
```
* 安裝運行所需套件
```shell
pip install -r requirements.txt
```
* 執行 generate_json.ipynb，把需要的資料拿出來並寫成 pyserini 可以使用的形式，可以看到多一個 corpus 資料夾。
* cd pyserini
* 在虛擬環境下 run build_index_sparse.sh 中的指令，可以看到多一個 indexes 資料夾。
* 運行 sparse.ipynb 後可以得到一份 rusult (jmlm_0.9999_token10.pkl)，這是透過 JMLM 這個搜尋方式下，可以找到的每個 session 的前一百名推薦結果
* 將 jmlm_0.9999_token10.pkl 拖入 ngrams 資料夾底下

## ngrams (執行時間約 5 分鐘)

* 執行 generate_n_gram_test.ipynb，會看到出現三個檔案(cfd_3grams_test、cfd_4grams_test、cfd_5grams_test)
* 執行 result.ipynb，得到要繳交的運算結果 ngrams_jmlm0.9999_token10.csv
* 成績:0.56505，感覺有哪裡出問題導致掉了一點分數

## 執行時間測試環境

* CPU:i5-12400
* 記憶體:64GB
* 記憶體空間不足的話可能會需要使用硬碟做為記憶體，會導致 ngrams 的運行速度大幅降低