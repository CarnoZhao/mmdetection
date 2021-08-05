python tools/test.py \
    ./config_detectors_50.py \
    ./work_dirs/pig/config_detectors_50_gc/latest.pth \
    --show-dir ./work_images \
    --show \
    --show-score-thr 0.3

tools/dist_test.sh ./config_detectors_50.py \
    ./work_dirs/pig/config_detectors_50_gc/latest.pth \
    4 \
    --format-only \
    --options "jsonfile_prefix=./work_dirs/pig/config_detectors_50_gc/test"