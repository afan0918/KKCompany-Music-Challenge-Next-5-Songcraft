# 如何運行

## pyserini

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

