# Facial emotion recognition (FER2013)

Classifies a face image into one of seven emotions (angry, disgust, fear,
happy, neutral, sad, surprise) using the
[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset of 48×48
greyscale faces.

Two approaches are trained and compared with fastai on PyTorch:

| Model | Accuracy |
|---|---|
| Custom CNN, defined from scratch in PyTorch | 52.7% |
| VGG19 (batch-norm), ImageNet-pretrained and fine-tuned | **68.6%** |

The fine-tuned VGG19 was kept and exported as `emotion_model.pkl`.

## Approach

- **Data:** FER2013's train/test folders, loaded with `ImageDataLoaders`, with
  augmentation: random resized crops (down to 20% scale), horizontal flips
  and rotations of up to 30°.
- **Class imbalance:** FER2013 is heavily skewed (e.g. very few "disgust"
  images), so training uses cross-entropy weighted by inverse class frequency.
- **Training:** one-cycle schedule for the custom model and `fine_tune` for
  VGG19, both with early stopping (patience 10) on accuracy.
- **Evaluation:** accuracy plus a confusion matrix per model.

Note: the test folder doubles as the validation set for early stopping, so
the reported accuracy is slightly optimistic. A held-out split would give a
cleaner figure.

## Files

- `trainer.ipynb`: data loading, both models, training and evaluation
- `tester.ipynb`: a small Gradio app that loads `emotion_model.pkl` and
  predicts the emotion in an uploaded image
- `emotion_model.pkl`: the exported fastai learner (VGG19)
- `fer2013.zip`: the dataset
