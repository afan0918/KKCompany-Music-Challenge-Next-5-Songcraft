cd ~/.virtualenvs/KKCompany-Music-Challenge-Next-5-Songcraft/bin
source activate
cd /mnt/c/Users/afan/Documents/GitHub/KKCompany-Music-Challenge-Next-5-Songcraft/final/pyserini
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input corpus/ \
  --index indexes/collection_jsonl_sparse \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeContents \
  --stemmer none \
  --keepStopwords