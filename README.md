Data preparation is at `./sartorius/data.ipynb`. Run this notebook under competition `data` directory. The default setting is to cut images into 16 smaller images, namely `train_tiny` (you should `mkdir -p train_tiny/images train_tiny/annotations` first).

Training config is saved at `work_configs/cell/cell_htc_r2101.py`. The default setting is training HTC-Res2Net-101 with fold0 data. You can change models, data as you wish. I trained with 2 TITAN RTX, with per GPU size 4, batch size 2x4=8.

Inferencing code can be found at [https://www.kaggle.com/carnozhao/cell-submission](https://www.kaggle.com/carnozhao/cell-submission). You should change ckpt path and config path to your own model when running inferencing.