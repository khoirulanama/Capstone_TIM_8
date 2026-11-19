# Hasil komparasi (dibangkitkan run_all.py)

| NIM | Model | seed | val_acc | val_macro_f1 | test_acc | durasi (s) | perangkat |
|---|---|---|---|---|---|---|---|
| baseline | softmax NumPy | 1 | 0.2488 | 0.2112 | — | 0 | cpu laptop |
| non_dl | random_forest | 1 | 0.3603 | 0.3250 | — | 1 | cpu laptop |
| 122450039 | cnn | 1 | 0.3689 | 0.2943 | — | 7 | cpu laptop |
| 122450041 | rnn | 1 | 0.2525 | 0.2132 | — | 307 | cpu laptop |
| 123450009 | trf | 1 | 0.2083 | 0.1210 | — | 9 | cpu laptop |
| 123450010 | ae | 1 | 0.3223 | 0.3010 | — | 38 | cpu laptop |
