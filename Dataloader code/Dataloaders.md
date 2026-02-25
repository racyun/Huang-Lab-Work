Working (slow) dataloader
https://colab.research.google.com/drive/1iSd0YS0tIU2tpC2n74zX212I6fW0utXs?authuser=1#scrollTo=6os6bgqvYH_V
Note: takes about 15 minutes to load a single sample - time we do not have!

Disked Cached Dataloader:
https://colab.research.google.com/drive/1cOpMmtcTCERwRA5texpiPffPubQr40p5?usp=sharing
Loads fast by saving images to the cache, but runs out of disk space early on as a result (crashes after loading ~40% of the data).
