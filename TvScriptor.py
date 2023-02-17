import numpy as np
import tensorflow as tf
import random
import sys
import io

"""_summary_
This script is a LSTM model that generates a script based on a given text file.
The model is trained on the first two seasons of The Office.
The model is then used to generate a script based on the first two seasons of The Office.
"""

# helper function to sample an index from a probability array
# This function is used to sample the next character in our script generation
# The higher the temperature, the more random the next character will be
# The lower the temperature, the more likely the next character will be the most likely character
# Temperature of 1.0 is the most diverse
# Temperature of 0.0 is the least diverse
def sample(preds, temperature=0.6):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# tells us what the character count is at the beginning of a specified line in our script
def charPerLine(my_text):
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

#define a constant variable for the file we want to write to    
file_write = io.open('files/generated_script.txt', mode='a' , encoding='utf-8' )  
print('corpus length:', len(text))
#variable for content to be written to the file
content = 'Text generated from The Office script'

# Create a mapping from unique characters to indices
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
dict_charPerLine = charPerLine(text)
content += ('\n' + 'corpus length: ' + str(len(text)) + '\n' + 'total chars: ' + str(len(chars)) + '\n')
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

content += ('\n' + 'nb sequences: ' + str(len(sentences)) + '\n' + 'Vectorization...' + '\n')
# Build the model: a single LSTM
print('Build model...')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(tf.keras.layers.Dense(len(chars)))
model.add(tf.keras.layers.Activation('softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
content += ('\n' + 'Build model...' + '\n')

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y,
              batch_size=128,
              epochs=1)
    
    #save the printed text to a file
    content += ('\n' + '-' * 50 + '\n' +' Iteration ' + str(iteration) )

    # Max(char_count_per_line.keys()) gives us the last line number of our script
    # However, what we want our seed to be is the value(), not the key
    # Thus, we can set sentence = char_count_per_line[start_index]
    # This ensures that our seed starts from a new line
    start_index = random.randint(0, max(dict_charPerLine.keys()))

    # Value can be changed to look at how diversity affects our generation
    # diversity in [0.2, 0.5, 1.0, 1.2] is a good range to look at
    for diversity in [0.3]:
        print()
        print('----- diversity:', diversity)
        #add the printed text to a file
        content += ('\n' + '----- diversity: ' + str(diversity) + '\n')

        generated = ''
        sentence = text[dict_charPerLine[start_index]: dict_charPerLine[start_index] + maxlen]
        generated += sentence
        print()
        print('----- Generating with seed: "' + sentence + '"')
        #save the printed text to a file
        content+=('\n')
        content+=('----- Generating with seed: "' + sentence + '"')
        
        
        sys.stdout.write(generated)
        content+=generated 
        
        # We generate 400 characters after the seed sentence to see how the model performs
        # The higher the temperature, the more random the next character will be
        # x_pred is the input to our model
        # preds is the output of our model
        # next_index is the index of the next character in our model
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            
            #save the generated text to a file
            content += (next_char)
            
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
        
#save the generated text to a file
file_write.write(content)
file_write.close()