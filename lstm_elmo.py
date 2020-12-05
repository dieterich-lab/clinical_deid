import argparse
import numpy as np
from elmoformanylangs import Embedder
import keras
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, Bidirectional, TimeDistributed
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D, Input, add
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import classification_report as classrep

# Defining arguments
parser = argparse.ArgumentParser(description='Train and evaluate a named entity model e.g. for de-identification. Clinical DeID.')

parser.add_argument('--path_train', type=str,
                    help='Path to the CoNLL formated training file.')

parser.add_argument('--path_test', type=str,
                    help='Path to the CoNLL formated test file.')


args = parser.parse_args()

path_train = args.path_train
path_test = args.path_test

# Defining the training and test data
#path_train = 'data/deid_surrogate_train_all_version2.conll'
#path_test = 'data/deid_surrogate_test_all_groundtruth_version2.conll'

# Create data set
def create_dataset(path, max_len=0):
    letters = []
    letter = []

    labels = []
    label = []
    tags = []
    words = []
    print('Extracting the text...')
    with open(path, 'r') as f:
        for line in f:
            line = line.split('\t')
            if len(line) > 1:
                letter.append(line[0].strip())
                label.append(line[1].strip())
                tags.append(line[1].strip())
                words.append(line[0].strip())
            else:
                letters.append(letter)
                labels.append(label)
                label = []
                letter = []

    print("Amount of words:")
    print(len(words))
    words = list(set(words))

    tags = list(set(tags))
    if max_len != 0:
        max_len = max_len
    else:
        max_len = max([len(s) for s in letters])


    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    tag2idx = {t: i for i, t in enumerate(tags)}
    print(tag2idx)
    y = [[tag2idx[w] for w in s] for s in labels]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

    new_X = []
    for seq in letters:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_X.append(new_seq)
    X = new_X

    print("Lenght of X:")
    print(len(X))
    return X, y, letters, words, max_len, tag2idx, tags

X_train, y_tr, letters_train, words_train, max_len, tag2idx, tags = create_dataset(path_train)
X_test, y_te, letters_test, _, _, _, _ = create_dataset(path_test, max_len=max_len)

print('Generating train ELMo embeddings...')

e = Embedder('configs/elmo')

'''
elmos_tr = e.sents2elmo(X_train)

X_tr = np.array(elmos_tr)

np.save('elmo_traini2b22006_eng',X_tr)
'''
X_tr = np.load('elmo_traini2b22006_eng.npy')

print('Generating test ELMo embeddings...')
'''
elmos_te = e.sents2elmo(X_test)

X_te = np.array(elmos_te)

np.save('elmo_testi2b22006_eng',X_te)
'''
X_te = np.load('elmo_testi2b22006_eng.npy')

# Creating character data set
def create_char_dataset(data, words, max_len):
    max_len_char = 15

    print('Generating character embeddings...')
    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)

    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0
    X_char = []
    for letter in data:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(letter[i][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    return X_char, max_len_char, n_chars

X_char_tr, max_len_char, n_chars = create_char_dataset(letters_train, words_train, max_len)
X_char_te, _, _ = create_char_dataset(letters_test, words_train, max_len)

idx2tag = {i: w for w, i in tag2idx.items()}

print('Training the model')
data_dim = 1024
timesteps = max_len
num_classes = len(tags)
batch_size = 128
epochs = 100

print(f"timesteps: {timesteps}")
print(f"num_classes: {num_classes}")
print(f"batch_size: {batch_size}")
print(f"X_char_tr.shape: {np.array(X_char_tr).shape}")
print(f"max_len_char: {max_len_char}")
print(f"len(y_tr): {len(y_tr)}")
print(f"max_len: {max_len}")
print(f"X_tr.shape: {X_tr.shape}")

word_in = Input(shape=(max_len,))


# Input and embedding for words
word_input = Input(shape=(timesteps,data_dim))

# input and embeddings for characters
char_in = Input(shape=(max_len, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=24,
                           input_length=max_len_char, mask_zero=True))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=64, return_sequences=False,
                                recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([word_input, char_enc])
x = SpatialDropout1D(0.3)(x)

x = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(x)

out = TimeDistributed(Dense(num_classes, activation="softmax"))(x)

model = Model([word_input, char_in], out)

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["acc"])

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='models/best_model_lstm_elmo_i2b22006.h5', monitor='val_loss', save_best_only=True)]

print(model.summary())

history = model.fit([X_tr,
                     np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                    np.array(y_tr).reshape(len(y_tr), max_len, 1),
                    batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1, callbacks=callbacks)


# Evaluation
test_pred = model.predict([X_te,
                        np.array(X_char_te).reshape((len(X_char_te),
                                                     max_len, max_len_char))])

# Entitywise classification report
y_test = [[idx2tag[t] for t in y_te[i]] for i in range(len(y_te))]
y_pred = [[idx2tag[t] for t in np.argmax(test_pred[i], axis=-1)] for i in range(len(test_pred))]

print(classification_report(y_test, y_pred))

# Tokenwise classification
y_test = np.array(y_test).flatten()
y_pred = np.array(y_pred).flatten()

tags.remove('O')

print(classrep(y_test, y_pred, labels=tags))


