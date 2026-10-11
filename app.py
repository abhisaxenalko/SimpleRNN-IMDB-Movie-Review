"""Streamlit app for training and using a SimpleRNN sentiment classifier."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.callbacks import TensorBoard


# Configure the browser tab before calling any other Streamlit commands.
st.set_page_config(page_title="IMDB SimpleRNN Sentiment", page_icon="🎬")

# Keep training data, model logs, and all relative paths beside this app.
APP_DIR = Path(__file__).resolve().parent
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_SEQUENCE_LENGTH = 200


# Download the standard IMDB dataset once per Streamlit cache key, then pad
# every review to the same length so it can be passed to the recurrent model.
@st.cache_data(show_spinner="Loading and preparing the IMDB movie-review dataset...")
def load_imdb_data(
    vocab_size: int, sequence_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    (train_reviews, train_labels), (test_reviews, test_labels) = imdb.load_data(
        num_words=vocab_size
    )
    word_index = imdb.get_word_index()

    padded_train = pad_sequences(
        train_reviews,
        maxlen=sequence_length,
        padding="post",
        truncating="post",
    )
    padded_test = pad_sequences(
        test_reviews,
        maxlen=sequence_length,
        padding="post",
        truncating="post",
    )

    return (
        padded_train,
        np.asarray(train_labels, dtype=np.int32),
        padded_test,
        np.asarray(test_labels, dtype=np.int32),
        word_index,
    )


# Build a binary sentiment model: an embedding learns word representations,
# SimpleRNN reads each padded review in sequence, and sigmoid gives a positive
# sentiment probability.
def build_rnn_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int,
    rnn_units: int,
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(sequence_length,), dtype="int32"),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=True,
            ),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Binary cross-entropy trains the positive/negative classifier; accuracy
    # is tracked in TensorBoard while the model learns.
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Convert typed review text into the same integer IDs used by Keras's IMDB
# dataset, including its start token, unknown-word token, and zero padding.
def encode_review(
    review: str,
    word_index: dict[str, int],
    vocab_size: int,
    sequence_length: int,
) -> np.ndarray:
    tokens = text_to_word_sequence(review)
    review_ids = [1]

    for token in tokens:
        dataset_index = word_index.get(token)
        model_index = dataset_index + 3 if dataset_index is not None else 2
        if model_index >= vocab_size:
            model_index = 2
        review_ids.append(model_index)

    return pad_sequences(
        [review_ids],
        maxlen=sequence_length,
        padding="post",
        truncating="post",
    )


# Present the app controls and explain the binary labels shown in predictions.
st.title("Movie Review Sentiment — SimpleRNN")
st.write(
    "Train a recurrent neural network on the Keras IMDB dataset, then enter "
    "a movie review to estimate whether its sentiment is positive or negative."
)

with st.sidebar:
    st.header("Training settings")
    vocab_size = st.slider("Vocabulary size", 5_000, 20_000, DEFAULT_VOCAB_SIZE, 1_000)
    sequence_length = st.slider(
        "Review sequence length", 100, 400, DEFAULT_SEQUENCE_LENGTH, 50
    )
    embedding_dim = st.slider("Embedding dimensions", 8, 64, 16, 8)
    rnn_units = st.slider("SimpleRNN units", 16, 128, 32, 16)
    epochs = st.slider("Training epochs", 1, 10, 3)

# Keep the user in control of the relatively expensive dataset download and
# training step. The SciKeras wrapper adapts the Keras model to an sklearn API.
if st.button("Train SimpleRNN", type="primary"):
    (
        x_train,
        y_train,
        x_test,
        y_test,
        word_index,
    ) = load_imdb_data(vocab_size, sequence_length)

    log_dir = APP_DIR / "logs" / "fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=str(log_dir))
    classifier = KerasClassifier(
        model=build_rnn_model,
        model__vocab_size=vocab_size,
        model__sequence_length=sequence_length,
        model__embedding_dim=embedding_dim,
        model__rnn_units=rnn_units,
        epochs=epochs,
        batch_size=128,
        verbose=0,
        random_state=42,
    )

    # Train against the official training split and reserve part of it for
    # validation; the untouched official test split is used for evaluation.
    with st.spinner("Training the SimpleRNN model..."):
        classifier.fit(
            x_train,
            y_train,
            validation_split=0.1,
            callbacks=[tensorboard_callback],
        )
        test_probabilities = classifier.predict_proba(x_test)

    # SciKeras returns one probability column per class; class 1 is positive.
    predicted_labels = np.argmax(test_probabilities, axis=1)
    test_accuracy = float(np.mean(predicted_labels == y_test))

    # Store the trained classifier and its matching preprocessing vocabulary
    # in the Streamlit session so later reruns can make predictions.
    st.session_state["classifier"] = classifier
    st.session_state["word_index"] = word_index
    st.session_state["vocab_size"] = vocab_size
    st.session_state["sequence_length"] = sequence_length
    st.session_state["test_accuracy"] = test_accuracy
    st.session_state["log_dir"] = str(log_dir)

# Report the test-set result and the TensorBoard log path for the active model.
if "classifier" in st.session_state:
    st.success("The model is trained and ready to classify reviews.")
    metrics = pd.DataFrame(
        [{"Evaluation split": "IMDB test set", "Accuracy": st.session_state["test_accuracy"]}]
    )
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.caption(f"TensorBoard logs: `{st.session_state['log_dir']}`")

    # Encode user text with the vocabulary and sequence length used at training,
    # then map the predicted probability to a human-readable sentiment label.
    review_text = st.text_area("Enter a movie review", height=160)
    if st.button("Predict sentiment"):
        if not review_text.strip():
            st.warning("Enter a review before requesting a prediction.")
        else:
            review_array = encode_review(
                review_text,
                st.session_state["word_index"],
                st.session_state["vocab_size"],
                st.session_state["sequence_length"],
            )
            probabilities = st.session_state["classifier"].predict_proba(review_array)
            # SciKeras returns a 1D class-probability array for one review, so
            # add the batch dimension before selecting the positive class.
            probabilities = np.atleast_2d(probabilities)
            positive_probability = float(probabilities[0, 1])

            sentiment = "Positive" if positive_probability >= 0.5 else "Negative"
            st.subheader(f"Sentiment: {sentiment}")
            st.write(f"Positive sentiment probability: {positive_probability:.1%}")
else:
    st.info("Choose training settings in the sidebar and select **Train SimpleRNN**.")
