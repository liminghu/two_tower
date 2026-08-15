# two_tower
Two Tower model tutorial and comparison with other models on goodbooks-10k data.

The goodbooks-10k dataset is a popular benchmark collection for building book recommendation systems. Created by Zygmunt Zając, it contains 6 million ratings for the 10,000 most popular books from 53,424 users. Ratings range from 1 to 5, and users have rated at least two books.Core Files & Structureratings.csv: Contains 6 million numerical user-to-book rating pairs (scaled 1 to 5) sorted chronologically. Both user and book IDs are contiguous integers (users: 1–53,424; books: 1–10,000).books.csv: Provides metadata for each of the 10,000 books, including authors, publication years, titles, and average Goodreads ratings.to_read.csv: Tracks approximately 1 million user-to-book interactions where users have marked books on their "to-read" shelves.book_tags.csv / tags.csv: Captures user-generated tags, genres, and shelf categories assigned to the books.Key CharacteristicsPopularity Bias: It focuses specifically on the top 10,000 most-rated books on the platform rather than obscure or long-tail titles.Data Density: Most books contain roughly 100 reviews, providing a dense interaction matrix well-suited for collaborative filtering and matrix factorization algorithms.

ease_goodbooks_anthropic.py
Users: 53424, Items: 10000
Train interactions: 5709394
Fitting EASE with lambda=100.0 ...
  lambda=100.0    val recall@20 = 0.4044
Fitting EASE with lambda=300.0 ...
  lambda=300.0    val recall@20 = 0.3988
Fitting EASE with lambda=700.0 ...
  lambda=700.0    val recall@20 = 0.3847

Best lambda: 100.0

Best validation recall@20: 0.4044


two_tower_goodbooks_anthropic.py
Using device: cuda
Downloading ratings.csv to data/ratings.csv ...

Users: 53424, Items: 10000, Train pairs: 5709394

epoch  1 | loss 6.1814 | val recall@20 0.1919

epoch  2 | loss 5.5486 | val recall@20 0.2405

epoch  4 | loss 5.2247 | val recall@20 0.2758

epoch  6 | loss 5.1160 | val recall@20 0.2880

epoch  8 | loss 5.0654 | val recall@20 0.2933

epoch 10 | loss 5.0357 | val recall@20 0.2973

epoch 12 | loss 5.0166 | val recall@20 0.2985

epoch 14 | loss 5.0030 | val recall@20 0.3000

epoch 16 | loss 4.9929 | val recall@20 0.2990

epoch 18 | loss 4.9854 | val recall@20 0.3024

epoch 20 | loss 4.9794 | val recall@20 0.3004

epoch 22 | loss 4.9750 | val recall@20 0.3025

epoch 24 | loss 4.9710 | val recall@20 0.3012

epoch 26 | loss 4.9677 | val recall@20 0.3025

epoch 28 | loss 4.9653 | val recall@20 0.3040

epoch 30 | loss 4.9630 | val recall@20 0.3029

Best validation recall@20: 0.3040


Two Tower Model demo:
Masked Recall@20 (Model): 0.1958
Masked Recall@20 (Pop)  : 0.0809
Lift (abs): +0.1150
Lift (rel): +142.15%

Two Tower Model Demo 2:
Masked Recall@20 (Model): 0.1980
Masked Recall@20 (Pop)  : 0.0809
Lift (abs): +0.1171
Lift (rel): +144.79%

python goodbooks_two_tower_pytorch.py \
  --ratings ratings.csv \
  --min-rating 3 \
  --min-user-interactions 10 \
  --min-item-interactions 10 \
  --dim 128 \
  --epochs 12 \
  --batch-size 4096 \
  --lr 1e-3 \
  --temperature 0.10 \
  --pop-bias-scale 2.0 \
  --target-recall 0.20

Epoch 01/12 loss=7.7626 time=39.0s
Validation masked Recall@20: 0.1689

Epoch 02/12 loss=7.0874 time=37.8s
Validation masked Recall@20: 0.2067

Epoch 03/12 loss=6.9097 time=37.9s
Validation masked Recall@20: 0.2293

Epoch 04/12 loss=6.8050 time=38.0s
Validation masked Recall@20: 0.2454

Epoch 05/12 loss=6.7323 time=38.7s
Validation masked Recall@20: 0.2586

Epoch 06/12 loss=6.6790 time=37.8s
Validation masked Recall@20: 0.2662

Epoch 07/12 loss=6.6381 time=38.6s
Validation masked Recall@20: 0.2731

Epoch 08/12 loss=6.6063 time=38.6s
Validation masked Recall@20: 0.2800

Epoch 09/12 loss=6.5813 time=38.4s
Validation masked Recall@20: 0.2835

Epoch 10/12 loss=6.5613 time=41.5s
Validation masked Recall@20: 0.2881

Epoch 11/12 loss=6.5452 time=47.1s
Validation masked Recall@20: 0.2908

Epoch 12/12 loss=6.5313 time=50.6s
Validation masked Recall@20: 0.2911

Best validation masked Recall@20: 0.2911

python goodbooks_two_tower_pytorch.py \
  --ratings ratings.csv \
  --min-rating 3 \
  --min-user-interactions 10 \
  --min-item-interactions 10 \
  --dim 192 \
  --epochs 12 \
  --batch-size 4096 \
  --lr 1e-3 \
  --temperature 0.08 \
  --pop-bias-scale 2.0 \
  --target-recall 0.35

Epoch 01/12 loss=7.6934 time=41.7s
Validation masked Recall@20: 0.1752

Epoch 02/12 loss=7.0255 time=39.8s
Validation masked Recall@20: 0.2193

Epoch 03/12 loss=6.8250 time=39.5s
Validation masked Recall@20: 0.2493

Epoch 04/12 loss=6.6953 time=39.9s
Validation masked Recall@20: 0.2669

Epoch 05/12 loss=6.6065 time=40.1s
Validation masked Recall@20: 0.2798

Epoch 06/12 loss=6.5433 time=40.1s
Validation masked Recall@20: 0.2894

Epoch 07/12 loss=6.4956 time=40.2s
Validation masked Recall@20: 0.2948

Epoch 08/12 loss=6.4576 time=40.1s
Validation masked Recall@20: 0.3006

Epoch 09/12 loss=6.4265 time=40.0s
Validation masked Recall@20: 0.3045

Epoch 10/12 loss=6.4007 time=40.0s
Validation masked Recall@20: 0.3095

Epoch 11/12 loss=6.3794 time=39.5s
Validation masked Recall@20: 0.3118

Epoch 12/12 loss=6.3616 time=40.0s
Validation masked Recall@20: 0.3140

Best validation masked Recall@20: 0.3140

python goodbooks_two_tower_pytorch.py --ratings ratings.csv
Epoch 01/12 loss=7.7626 time=40.2s
Validation masked Recall@20: 0.1689

Epoch 02/12 loss=7.0874 time=37.6s
Validation masked Recall@20: 0.2067

Target reached: Recall@20 >= 0.20
Best validation masked Recall@20: 0.2067


python goodbooks_two_tower_pytorch.py \
  --ratings ratings.csv \
  --min-rating 3 \
  --min-user-interactions 10 \
  --min-item-interactions 10 \
  --dim 192 \
  --epochs 20 \
  --batch-size 4096 \
  --lr 1e-3 \
  --temperature 0.08 \
  --pop-bias-scale 2.0 \
  --target-recall 0.35

Epoch 02/20 loss=7.0255 time=39.6s
Validation masked Recall@20: 0.2193

Epoch 03/20 loss=6.8250 time=39.5s
Validation masked Recall@20: 0.2493

Epoch 04/20 loss=6.6953 time=39.3s
Validation masked Recall@20: 0.2669

Epoch 05/20 loss=6.6065 time=39.6s
Validation masked Recall@20: 0.2798

Epoch 06/20 loss=6.5433 time=39.4s
Validation masked Recall@20: 0.2894

Epoch 07/20 loss=6.4956 time=39.5s
Validation masked Recall@20: 0.2948

Epoch 08/20 loss=6.4576 time=39.5s
Validation masked Recall@20: 0.3006

Epoch 09/20 loss=6.4265 time=39.4s
Validation masked Recall@20: 0.3045

Epoch 10/20 loss=6.4007 time=39.5s
Validation masked Recall@20: 0.3095

Epoch 11/20 loss=6.3794 time=39.1s
Validation masked Recall@20: 0.3118

Epoch 12/20 loss=6.3616 time=39.3s
Validation masked Recall@20: 0.3140

Epoch 13/20 loss=6.3460 time=39.2s
Validation masked Recall@20: 0.3164

Epoch 14/20 loss=6.3329 time=39.4s
Validation masked Recall@20: 0.3197

Epoch 15/20 loss=6.3212 time=39.1s
Validation masked Recall@20: 0.3201

Epoch 16/20 loss=6.3106 time=39.6s
Validation masked Recall@20: 0.3193

Epoch 17/20 loss=6.3014 time=39.4s
Validation masked Recall@20: 0.3208

Epoch 18/20 loss=6.2932 time=39.7s
Validation masked Recall@20: 0.3235

Epoch 19/20 loss=6.2854 time=39.4s
Validation masked Recall@20: 0.3247

Epoch 20/20 loss=6.2786 time=39.2s
Validation masked Recall@20: 0.3232

Best validation masked Recall@20: 0.3247  

  
