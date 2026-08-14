# two_tower
Two Tower model tutorial and comparison with other models on goodbooks-10k data.


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
