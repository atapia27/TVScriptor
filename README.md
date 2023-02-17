# TVScriptor
*A Python tool for generating compelling television scripts using recurrent neural networks, natural language processing, and deep learning text generation techniques.*


**_Output generated from TV Show, The Office:_**

```
corpus length: 479620
total chars: 84
nb sequences: 159860
Vectorization...
Build model...
--------------------------------------------------
Iteration 1
1249/1249 [==============================] - 126s 97ms/step - loss: 1.9933

----- diversity: 0.3

----- Generating with seed: "Kevin: That sucks so much.
Guy: It total"
Kevin: That sucks so much.
Guy: It totall to suest to with to to to to to to be to to be do to to to to to to to to to to to to to tell to to to be to we was whing about I was the was work to to be a to to to to to so like to to to to to to to be to to to to to be to to be to the be sure to be to me to to to to to to be for there to to to to to do no worked to to to to to to to to to to to to be to to to to to the a to to to to to to to

--------------------------------------------------

Iteration 5
1249/1249 [==============================] - 82s 66ms/step - loss: 1.3657

----- diversity: 0.3

----- Generating with seed: "Jim: [on phone] Hey, Brenda. This is, uh"
Jim: [on phone] Hey, Brenda. This is, uh, the comples to the romer and the fample and I am a bumpries to me see your for beer, I think the warnand is the promise. I got to you all go an in the fine. I am the coming a lot of the fine them.
Michael: What? What is a little back. There is a second and I have a lot of you things to sering on the cool. And I was going to be the really some on the cards and I was
 that they would stay thing to 


--------------------------------------------------
Iteration 10
1249/1249 [==============================] - 159s 128ms/step - loss: 1.2627

----- diversity: 0.3

----- Generating with seed: "Dwight: God... Damn it! Why us?
Jim: Bec"
Dwight: God... Damn it! Why us?
Jim: Because I know that they are the neckeds.
Jim: And the can getting a not show that you know it's a serodle and they need to be good. Oh, all right, we're going to talk to you a county.
Michael: Okay. What is the real the corneard.
Michael: What is the things that they are the real to the the planng and the corporate the corporate the personal in the books at me this good birthday. It was a person and

--------------------------------------------------
Iteration 15
1249/1249 [==============================] - 385s 308ms/step - loss: 1.2164

----- diversity: 0.3

----- Generating with seed: "Dwight: Forget about retiring when you'r"
Dwight: Forget about retiring when you're to the back on. I have to see you to take the takes of the say there.
Dwight: I can't get to take the the back of the thing about the health girl. Yeah, you know what? I can't do the takes here.
Michael: Yeah.
Michael: Oh, no, no, no, no, no, no, no, no. I don't know what? I have to be a looking the thing about.
Michael: I mean, I was thinking at the plan back have been the look at the booter of

--------------------------------------------------
Iteration 20
1249/1249 [==============================] - 452s 362ms/step - loss: 1.1849

----- diversity: 0.3

----- Generating with seed: "Dwight: What are you doing? Those are my"
Dwight: What are you doing? Those are my friends someone works in the contact of the thing that was a bottle about the bord the thing are not my friends.
Michael: Okay, well, I was a second on the bother of my drink?
Michael: Okay. That was a groustand Michael and then you are it and then I was a speciate the fire of the cares. I don't have the bother beerstand.
Michael: Oh, I think that was a fire of the still be the carples.
Michael: 

--------------------------------------------------
Iteration 21
1249/1249 [==============================] - 517s 414ms/step - loss: 1.1820

----- diversity: 0.3

----- Generating with seed: "Dealer: Flip them.
Michael: You really s"
Dealer: Flip them.
Michael: You really should not and you would have a good part to the card. That's a little things. I don't have a tool cold an ince the part to the compried. [clapping and the day.
Pam: I don't know.
Michael: Yeah, uh, this is a second of me and they can be the real about the compares.
Michael: Oh, hey, uh, what I moved it to be anyone.
Michael: That was a point and then I was a scits of Michael takes hands and so I w

--------------------------------------------------
Iteration 22
1249/1249 [==============================] - 514s 412ms/step - loss: 1.1757

----- diversity: 0.3

----- Generating with seed: "Dwight: First present, Oscar.
Oscar: [ri"
Dwight: First present, Oscar.
Oscar: [right to the confererring] What is the receptionist.
Jim: Yeah, I think you are going to be the first year, you know what? I am a good branch of the receptionis. I will be the office, and I would have something the big stand of the takes here to go to do the back branch of the company of the conterchinating out the office.
Dwight: We are going to say I will be the receptionion of the really because 

--------------------------------------------------
Iteration 23
1249/1249 [==============================] - 465s 372ms/step - loss: 1.1725

----- diversity: 0.3

----- Generating with seed: "Michael: Oh, [taking it to heart] lazy. "
Michael: Oh, [taking it to heart] lazy. I won't stay and I don't know.
Ryan: Ok, I just want to see you the talked the back at the seats of here all bet me when I was a little the complaint. Okay, I was a birthday of the it works because you
 want to be the sent of the seats and I don't know what I don't have a little find of the complete the big assed the seats of manager. So you want to be a really sorry.
Michael: I don't know. I want 

--------------------------------------------------
Iteration 24
1249/1249 [==============================] - 474s 380ms/step - loss: 1.1693

----- diversity: 0.3

----- Generating with seed: "Ryan: Uh, hanging out with some friends,"
Ryan: Uh, hanging out with some friends, and they are the person around of the conternity.
Michael: [sings around the thing at the wher in a person around the complaines a bott. And they are the seats and they would be a lot of her of the parties.
Michael: Okay, here it is a bump to the fore the contoner. The harge a lot of the conterrith start the contact of the leaver. That's a thing and they are a good busy of the corporace and they 

etc. ...
```

>*This code is using a LSTM (Long Short-Term Memory) neural network to perform text generation. It does this by training the model on a given text corpus and using the trained model to generate new text by predicting the next character given a sequence of characters from the text. The model is trained using the TensorFlow library and the text is vectorized and processed using the numpy and io libraries. The generated text is created by sampling from the predicted probability distribution of the next character, with a higher temperature resulting in more randomness in the generated text. The model is trained for a certain number of iterations and the generated text is outputted after each iteration.*

>*LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that is specifically designed to remember patterns over long periods of time. RNNs are a type of neural network that are designed to process sequential data, such as time series, natural language, or music. LSTMs are a variant of RNNs that have a more complex architecture, which allows them to learn longer-term dependencies and avoid the vanishing gradient problem.*

>*In this code, the diversity value is a hyperparameter used in the text generation process. It determines the "creativity" or "randomness" of the generated text by controlling the temperature of the softmax function in the model's output layer.*

>*Vectorization is the process of converting data into a numerical form that can be processed by a machine learning model. In the context of this code, the input data is a sequence of characters and the output data is a single character. These data are represented as arrays of boolean values, where each position in the array corresponds to a unique character in the text.*

>*The input data is stored in a 3D numpy array called x, and the output data is stored in a 2D numpy array called y. The array x has dimensions (len(sentences), maxlen, len(chars)), where len(sentences) is the number of sequences, maxlen is the length of each sequence, and len(chars) is the number of unique characters in the text. The array y has dimensions (len(sentences), len(chars)), where len(sentences) is the number of sequences and len(chars) is the number of unique characters in the text.*
