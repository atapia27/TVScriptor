import numpy as np
import tensorflow as tf
import random
import sys
import io


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# tells us what the character count is at the beginning of a specified line in our script
def characters_per_line(my_text):
    line_char = {}
    counter = 1
    curr_char_sum = 0
    # split the string into lines
    lines = my_text.split('\n')

    # iterate through the lines
    for line in lines:
        line_char[counter] = curr_char_sum
        curr_char_sum += len(line)+1
        counter += 1

    return line_char


# Load the text file
with io.open('files/The_Office_First_Two_Seasons.txt', mode='r', encoding='utf-8') as f:
    text = f.read()
print('corpus length:', len(text))

# Create a mapping from unique characters to indices
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
line_char_dict = characters_per_line(text)

# Cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# Build the model: a single LSTM
print('Build model...')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(tf.keras.layers.Dense(len(chars)))
model.add(tf.keras.layers.Activation('softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    # Max(char_count_per_line.keys()) gives us the last line number of our script
    # However, what we want our seed to be is the value(), not the key
    # Thus, we can set sentence = char_count_per_line[start_index]
    # This ensures that our seed starts from a new line
    start_index = random.randint(0, max(line_char_dict.keys()))

    # Value can be changed to look at how diversity affects our generation
    for diversity in [0.3]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[line_char_dict[start_index]: line_char_dict[start_index] + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()