# IMDB SimpleRNN Sentiment App

This Streamlit application trains a TensorFlow `SimpleRNN` on the built-in
Keras IMDB movie-review dataset and predicts whether entered reviews express
positive or negative sentiment. SciKeras provides the scikit-learn-compatible
training wrapper, and TensorBoard records training metrics.

## Setup

Use a Python version supported by the TensorFlow release installed in your
environment. From this folder, install the dependencies and launch the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first training run downloads the IMDB dataset and word index. Choose the
model settings in the sidebar, then select **Train SimpleRNN**. Training logs
are written under `logs/fit/`.

To inspect the training curves in another terminal, run this from this folder:

```powershell
tensorboard --logdir logs/fit
```
