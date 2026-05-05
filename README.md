# CNN Baseline Reproduction – CIFAR-10

A reproducible CNN baseline for CIFAR-10 image classification with a focused **ablation study on optimizer selection and generalization**.

The project conducts controlled experiments comparing **SGD, Adam, AdamW, and RMSprop** under identical settings.  
SGD with momentum achieved the best validation accuracy (**80.53%**) and demonstrated superior generalization.

All experiments use fixed seeds, standardized preprocessing, and a constant learning rate to isolate optimizer behavior.

https://d1shasaini.hashnode.dev/my-baseline-broke-what-an-18-accuracy-gap-taught-me-about-deep-learning?utm_source=hashnode&utm_medium=feed

## Run
```bash
pip install torch torchvision numpy matplotlib scikit-learn
python src/train.py
