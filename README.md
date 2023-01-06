# TVScriptor
**A Python tool for generating compelling television scripts using recurrent neural networks, natural language processing, and deep learning text generation techniques.**

This code is using a LSTM (Long Short-Term Memory) neural network to perform text generation. It does this by training the model on a given text corpus and using the trained model to generate new text by predicting the next character given a sequence of characters from the text. The model is trained using the TensorFlow library and the text is vectorized and processed using the numpy and io libraries. The generated text is created by sampling from the predicted probability distribution of the next character, with a higher temperature resulting in more randomness in the generated text. The model is trained for a certain number of iterations and the generated text is outputted after each iteration.

Our data comes from the first two seasons of the popular TV show, The Office.
