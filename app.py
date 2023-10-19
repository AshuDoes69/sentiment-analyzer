import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Emotion_classify_Data.csv')
 
# Check the dataset and column names
print(data.head())
print(data.columns)



# Preprocess the data
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(data['Comment'].values)
X = tokenizer.texts_to_sequences(data['Comment'].values)
X = pad_sequences(X)

# Split the data into training and testing sets
Y = pd.get_dummies(data['Emotion']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the neural network model
# Change the output layer to have 3 units (assuming 3 unique emotions)
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))  # Adjust to 3 output units
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
import matplotlib.pyplot as plt
history = model.fit(X_train, Y_train, epochs=10, batch_size=180)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#accuracy of training data
print("Accuracy of training data")
print(model.evaluate(X_train,Y_train))
#accuracy of testing data
print("Accuracy of testing data")
print(model.evaluate(X_test,Y_test))

# Save the model
model.save('model.h5')

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))
       
 #prediction what kind of emotion is in -i seriously hate one subject to death but now i feel reluctant to drop it
text = input("Enter your comment: ")
text = [text]
tokenizer = Tokenizer(num_words=5000, split=' ')
text = tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen=63)
sentiment = model.predict(text)


import numpy as np
print(np.argmax(sentiment))

# Define the emotions
emotions = ['Fear' , 'anger','joy']

# Define a function to preprocess the user input
def preprocess_input(input_text):
    # Convert the input to lowercase
    input_text = input_text.lower()
    # Tokenize the input
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([input_text])
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]
    # Pad the input sequence
    max_sequence_length = 63
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences([input_sequence], maxlen=max_sequence_length, padding='post', truncating='post')
    return input_sequence

# Define a function to predict the emotion
def predict_emotion(input_text):
    # Preprocess the input
    input_sequence = preprocess_input(input_text)
    # Make the prediction
    prediction = model.predict(input_sequence)
    # Get the index of the predicted emotion
    predicted_index = np.argmax(prediction)
    # Get the predicted emotion
    predicted_emotion = emotions[predicted_index]
    return predicted_emotion

# Test the function
running = True
while running:
    input_text = input("Enter your comment: ")
    if input_text == 'quit':
        running = False
        break
    predicted_emotion = predict_emotion(input_text)
    print(predicted_emotion)


