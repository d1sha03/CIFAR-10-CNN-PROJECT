# CNN Baseline Reproduction – CIFAR-10

A reproducible CNN baseline for CIFAR-10 image classification with a focused **ablation study on optimizer selection and generalization**.

The project conducts controlled experiments comparing **SGD, Adam, AdamW, and RMSprop** under identical settings.  
SGD with momentum achieved the best validation accuracy (**80.53%**) and demonstrated superior generalization.

All experiments use fixed seeds, standardized preprocessing, and a constant learning rate to isolate optimizer behavior.

📄 A **detailed experimental report and ablation study** is available in the `docs/` directory.

## Run
```bash
pip install torch torchvision numpy matplotlib scikit-learn
python src/train.py
