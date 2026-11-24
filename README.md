# Next Word Prediction with LSTM

A next-word prediction project built with TensorFlow/Keras and served through a Streamlit web app. The model is trained on Shakespeare's *Hamlet* and predicts the most likely word that follows a given phrase.

## Files

| File | Description |
| --- | --- |
| `app.py` | Streamlit app that loads the trained model and predicts the next word |
| `exp.ipynb` | Notebook containing data preparation, model training, evaluation, and export |
| `hamlet.txt` | Local copy of the *Hamlet* text used for training |
| `next_word_lstm.h5` | Trained Keras model (HDF5 format) |
| `tokenizer.pickle` | Serialized Keras tokenizer with the word-to-index mapping |
| `requirements.txt` | Python dependencies |

## How it works

1. The text of *Hamlet* is loaded from the NLTK Gutenberg corpus (a local `hamlet.txt` copy is also included).
2. A Keras `Tokenizer` converts words to integer indices, and overlapping n-gram sequences are generated as training examples.
3. Sequences are pre-padded to a fixed length. 80% of the data is used for training and 20% for validation.
4. The model is a `Sequential` network with:
   - an `Embedding` layer (100 dimensions),
   - `LSTM(150, return_sequences=True)` with `Dropout(0.2)`,
   - `LSTM(100)`,
   - a `Dense` softmax output layer over the full vocabulary.
5. It is compiled with `categorical_crossentropy` loss and the `adam` optimizer, then saved to `next_word_lstm.h5` along with the tokenizer.

At prediction time, the input phrase is tokenized, truncated to the maximum sequence length if necessary, pre-padded, and passed to the model. The index with the highest probability is mapped back to a word.

> Note: although the Streamlit page title mentions early stopping, the notebook trains the saved model for a fixed 50 epochs. Validation loss starts increasing after the early epochs, so adding a Keras `EarlyStopping` callback is a possible improvement.

## Setup

```bash
pip install -r requirements.txt
```

The app expects the model and tokenizer files to be present in the same directory as `app.py`.

## Run the app

```bash
streamlit run app.py
```

Enter a sequence of words (the default example is `To be or not to`) and click **Predict Next Word** to see the model's prediction.

## Requirements

See `requirements.txt`. Main dependencies are TensorFlow, Streamlit, NumPy, and NLTK (plus scikit-learn, pandas, Matplotlib, TensorBoard, and scikit-keras used in the notebook).
