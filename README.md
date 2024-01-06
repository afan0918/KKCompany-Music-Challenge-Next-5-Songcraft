# KKCompany-Music-Challenge-Next-5-Songcraft

此競賽位於 https://www.kaggle.com/competitions/datagame-2023

我們的方案排名：1/172

最終設計了一個不會特別推薦熱門歌曲和運算速度自認為還算快的模型，方案中無任何機器學習方案和隨機項，依據統計學組成整個模型架構。

純靠 n grams 架構可以得到 0.55569 的分數(可以推估 70 萬首結果中的 50 萬首），加上 Jelinek-Mercer 語言模型可以將分數衝到 0.56580。

## 賽後
* 寫出了 result_after_game.ipynb
* Score: 0.56580 -> Score: 0.56691
* 後續應該還能引入機器學習去動態評定優先級，但我就很懶惰啊，感覺測資要搞很久。

## 解題思路

主要問題是我們應該要怎麼有效去根據 25 首歌的聆聽順序去知道一首歌有多大的可能性出現在用戶的聆聽序列裡面?

大家都知道 n gram 會很好，但在冷門歌曲會怎麼樣?

我在數據集中隨便找了('09afc5a29b686b0dfff1a3a5317398d4', 'ab87181893acfc80a465c39a443ea08b')這筆數據，如果採用單純的n grams的話，會得到 
```
{'f223a183a28418c403c63955d79d9bda': 1, 
 '56cf3ecdd90729f015369c98151eb19b': 1, 
 'b26327d403a1101f32effb959d0128cd': 1} 
```
這樣的資料結構，老實說可以很清楚的看到n grams除非是遇到歌單，不然在冷門歌曲的推理成功率超級低，因為這三首歌的播放關係要如何去決定?要選哪一首其實猜到用戶的偏好難度都很高。

我們的著手點於 n grams 既然能夠很好的猜測歌曲，那我們能夠怎麼更好的捕捉用戶聆聽歌曲的模糊關係，我們採用了一個很簡單的方式來計算加權，採用dcg方式進行給分是因為我們定義了一個簡單的獎勵函數來對這次比賽進行貼近，但只要有這個結構，其實就算都給同樣的分數效果也會很好。

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17875609%2F4cbac723edcf04455f67ff0aeeedca84%2Fimage.png?generation=1704536761301711&alt=media)

這樣做的好處是什麼，可以在我們找到的內容中看到。
```
[('56cf3ecdd90729f015369c98151eb19b', 2.13),
 ('f223a183a28418c403c63955d79d9bda', 1.63),
 ('b26327d403a1101f32effb959d0128cd', 1),
 ('7ba90df9bff5c864d5f89d6fdc36b664', 0.63),
 ('56ac6c4b6fb9b5a38dafdacaa718cb4e', 0.5),
 ('1858e0efb54fd28fd852355eef289288', 0.5),
 ('9b1683a0af39085ec55fe706506518eb', 0.43),
 ('5bec1696b6167510be10e389f3b2b2d7', 0.43),
 ('f1d9e6c5188a9b6c862704e4f4ef7e96', 0.38),
 ('9dcdc0dbd95d76bc98b938a1c4e2552f', 0.38)]
```
只要使用者聽過('09afc5a29b686b0dfff1a3a5317398d4', 'ab87181893acfc80a465c39a443ea08b')這個順序，56cf3ecdd90729f015369c98151eb19b這首歌曲分別在他後面第一、二、三首被播放。

而f223a183a28418c403c63955d79d9bda在後面第一二首被播放，這說明了在清單之外，使用者聆聽的歌曲其實是有模糊的先後順序關係的，而我們的結構在這部分可以直觀的捕捉。

我們認為只要蒐集的歌曲樣本夠多，就可以簡易的拼湊出用戶可能會喜歡的歌曲和進行推薦，而在 n grams 中，這樣的模糊對應關係其實是會被吃掉的，所以這是我們的 n grams 在冷門歌曲上也會有一定成效的關係，跳脫猜測歌曲清單的範疇，我們做的比較像是把機器學習要學的東西直接拿出來做估計。

但是 n grams 雖然很好，可是只有一半的歌曲能用，剩下的歌曲應該怎麼辦?

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17875609%2Fab8b1233337ce7abace3410df20d8985%2Fimage.png?generation=1704537409799896&alt=media)

我們使用了 Language Models with Jelinek-Mercer Smoothing，將 100 萬首歌曲當成是 100 個 document，而歌曲的meta data(歌手、專輯、曲風、語言)和聆聽這首歌的 session 組成 document 的內容來進行傳統的網路搜尋行為並做調參。

因為我們發現熱門的歌曲在推薦熱門歌序列的時候很有效，但在冷門歌曲推薦上，我們發現著重在全局統計上，調低熱門歌曲的影響力會效果更好，另外使用這個方法可以讓推薦系統在不重覆推薦用戶聽過的歌曲的情況下，達到比直接推薦後五首歌或random更好的效果，最終來對使用者進行有效推薦。

實務上的資料我們也對新歌曲做了很多考量，在簡報的時候我簡單提了一個每天把舊數據乘以 0.98 的行為，但好像沒人聽懂可以對歌曲產生多大的影響，這代表可以透過低成本的方式每天迭代資料集，原本的聆聽行為過了30天就會剩下一半的影響力，權重低於一定值也可以把歌曲踢出模型，讓模型不會無限變大。

我們認為這樣會比深度學習更好來做調控，用戶自己進行的點閱也可以透過放大權重的方式來快速讓模型轉變，但統計模型聽起來就沒有那麼潮，在解釋我們模型運作的時候，如果沒有一直摸，感覺大家也很難馬上理解有多厲害，所以就，好吧。

## 賽後碎碎念

我們認為我的模型才是最接近商用的，因為後五首歌曲會大量跟第19、20首歌重複，所以其他組基本上都會ngrams找不到就直接填。

但我們覺得這樣根本是不能實用的，透過 Language Models with Jelinek-Mercer Smoothing 做到不用推薦用戶聆聽過的歌曲還可以找到冷門用戶感興趣的歌曲造成效能提升，但評審好像不覺得最後五首歌直接填重複的推薦出去有什麼關係。

另外我們的 n grams 定義其實跟其他 n grams 定義不太一樣，所以才能天然比其他 n grams 找到更多東西，其他組的 n grams 其實只適合拿來推斷 NLP 模型，但我們認為沒有那麼適合拿來做歌曲推薦，我們的研究脈絡其實是因為 GNN 效果沒那麼好，乾脆把 GNN 學習的機率直接抽出來成為一個簡單的統計模型。

有些組的 n grams 目的是為了抄歌單所以一路選擇最大機率往後推，但是其實冷門歌曲他很大可能接下來的所有歌曲只有聽一到兩次，根本沒有在歌單裡面，要怎麼捕捉這種模糊行為其實才是我們魔改的核心，然後分數根本沒有優勢，感覺做的東西好像被覺得很容易。

還有其他人的相似度模型基本上是直接比較歌曲相似度，我們為了減少歌曲運算量和系統負荷，所以只取前面十個其他用戶聆聽紀錄就可以找到比直接比較更好的冷門歌曲搜尋結果。

0.56-0.48~0.56-0.55就是我們的模型能比別人多找到的東西。

然後為了能夠上真實系統集群運算，我們的 n grams 模型其實是手刻的，原本有預備算力不夠的時候直接把 n grams 拆成幾個獨立的字典來分別跑，Language Models with Jelinek-Mercer Smoothing 也為了節約算力只需要儲存最接近的 10 個 session 就好。

不同的計算也是使用不同筆記本，因為一天有二十次提交機會，所以我是拿筆電和桌機分別算不同的部份，同時迭代不同細節，才能一天出二十份結果和測不同的東西。

最後，實務上 NN 需要成本，Google search 也是 使用 Language model + 網頁之間互相 link 計算而來的，其實我不認為 NN 能學到什麼東西，因為這次沒有那麼多資料，要預測一百萬首歌，根據經驗至少需要每首歌被聆聽一百次以上的樣本才能比較好的做學習，然後樣本其實是不均勻的，我們有試過幾種set、遮罩等等從統計學去模擬模型訓練的狀況來去探討為什麼機器學習訓練不起來，另外我不認為通用語言架構能夠適用於歌曲。

舉例來說，在通用語言模型中，「今天早上我要和朋友出去玩」和「朋友要和我今天早上出去玩」是同樣的意思，所以在訓練的時候，詞彙順序性不高。

但在聆聽行為上，聽完周杰倫再點去聽K-pop和聽完K-pop再點去周杰倫，用戶希望能得到的推薦清單其實是完全不同的，所以導致一些其他領域可以使用的機器學習模型會在這裡不準又消耗大量資源（雖然這裡感覺有點被很費資源拉走注意力的感覺）

這也是最後我們把架構砍到剩下兩個的原因，扣掉歌單，有出現過相似順序其實是可以根據分數去計算用戶偏好的，沒有出現過相似順序，也可以透過基礎語言模型去推斷哪些歌曲的用戶群是相近的，而不是透過集成模型的方式，我們覺得我們的模型最後簡化到可解釋性非常高。

但就沒辦法，我們把簡報從期末報告的20分鐘刪到只剩下我們的算法和一點實驗結果了，我們又想說運行速度很重要所以最後只有兩個系統來做，哇哇，希望下次最高成績至少有個獎。

## Contributors

|組員|貢獻內容|github|
|-|-|-|
|陳繹帆| ngrams+JMLM 解決方案研究 | [afan0918](https://github.com/afan0918) |
|陳品絜| MLE+pyserini 解決方案研究 | [pj-99](https://github.com/pj-99)|
|徐韶汶| EDA分析 | [AngelaHsu02](https://github.com/AngelaHsu02) |
|沈欣柏| 深度學習研究 | [jasonshen-python](https://github.com/jasonshen-python) |

## 運行前置動作（重要）

資料集不外流，請自行到 kaggle 下載後，將 datagame-2023 資料夾放置在本專案下，之後的具體運行請參閱 final 資料夾內的 README.md

## 運行環境

* Ubuntu 22.04(X86), python 3.10.12
* MacBook Pro(X86), python 3.10.12

其他環境下不保證能夠運行，但可以嘗試

## Overview
歡迎參加我們的音樂推薦競賽！這個競賽旨在挑戰您的機器學習和數據分析技能，以改進音樂串流體驗。您的任務是預測用戶在同一個聆聽 session 內聆聽一定數量歌曲後，接下來可能會聆聽哪些歌曲。這將有助於打造更個性化的音樂推薦服務，提高用戶滿意度。我們特別強調參賽者需要謹慎避免過度集中於熱門音樂，以確保推薦結果更多元化，滿足不同用戶的需求。期待參賽者的創新方法，以改進音樂串流體驗，提高用戶滿意度。

Welcome to our Music Recommendation Competition! This competition is designed to challenge your machine learning and data analysis skills to enhance the music streaming experience. Your task is to predict what songs users might listen to after hearing a certain number of songs in the same listening session. This will contribute to creating a more personalized music recommendation service, improving user satisfaction. We particularly emphasize that participants need to avoid an excessive focus on popular music to ensure diverse recommendation results. We look forward to innovative approaches from participants to enhance the music streaming experience and increase user satisfaction.

## Description
在這場競賽中，您將獲得用戶的歌曲播放記錄和歌曲相關資訊的數據集。您需要建立一個預測模型，該模型基於用戶在同一個聆聽 session 內聆聽的前 N (=20) 首歌曲，預測接下來會聆聽哪 K (=5) 首歌曲。我們將使用 DCG（Discounted Cumulative Gain）和預測歌曲的覆蓋率（Coverage）來評估模型的性能。請注意，避免過度集中於熱門歌曲是本競賽的一個關鍵要求。參賽者可以使用任何合法外部資源和工具，以開發創新的解決方案，以改進音樂串流體驗，提高用戶滿意度。

In this competition, you will receive a dataset containing user song playback records and song-related information. Your task is to build a predictive model that, based on the first N=20 songs played within the same listening session, forecasts the subsequent listening choices. We will evaluate the model's performance using DCG (Discounted Cumulative Gain) and coverage. Please note that avoiding an excessive focus on popular songs is a key requirement of this competition. Participants are allowed to use any legitimate external resources and tools to develop innovative solutions to enhance the music streaming experience and increase user satisfaction. We look forward to seeing your submissions in this competition.

