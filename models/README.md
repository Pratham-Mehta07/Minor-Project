# models/

Model weights are **not committed to git** (see `.gitignore`).

| Path | Produced by | Used by |
|------|-------------|---------|
| `models/pretrained/yolov8*.pt` | Downloaded automatically by Ultralytics (COCO-pretrained starting weights) | `ml/train.py`, `ml/check_env.py` |
| `models/best.pt` | Promoted from a training run (`ml/train.py --promote`) | Backend (`YOLO_MODEL_PATH`), `ml/predict.py`, `ml/evaluate.py` |
| `models/best.json` | Written next to `best.pt` on promotion: model version, source run, training config, dataset info | Backend stores the version with every analysis |

Raw training outputs (`best.pt`, `last.pt`, plots, metrics, `args.yaml`) stay in
`runs/detect/<experiment-name>/`. Only an explicitly promoted model is copied here,
so a new training run can never silently replace the model the API is serving.

If you train on Colab/Kaggle, download `runs/detect/<experiment-name>/weights/best.pt`
and place it at `models/best.pt`, or run the promote step there and download both
`best.pt` and `best.json`.
