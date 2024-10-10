#  Twitter Sentiment Analysis
#  Kaggle Dataset link : https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
# In this notebook we are going to do 

# 1. Importing the Dataset
# 2. Differentiating Labels and Features
# 3. Text Preprocessing 
# 4. Label Encoding 
# 5. Tokenization
# 6. LSTM model architecture
# 7. model compilation and traininig
# 8. metrics visualization


#  Functions to convert the data into lowercase , and remove its punctuations
def lower_case_and_remove_punctuation(arr):
    def process_string(s):
        if isinstance(s, str):
            s = s.lower()
            s = re.sub(r'[^\w\s]', '', s)
        return s

    process_func = np.vectorize(process_string)
    return process_func(arr)


#  Importing the Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

#  Training Dataset
df = pd.read_csv(r"your training path ")  # Put your training dataset over here
feature = df.iloc[:,-1].values
label = df.iloc[:,-2].values

# Validation Dataset
val_df = pd.read_csv(r"your testing path ") # relatively testing here 
val_feature = val_df.iloc[:,-1].values
val_label = val_df.iloc[:,-2].values

feature = lower_case_and_remove_punctuation(feature)

# Encoding the Labels into Numeric Values
LE = LabelEncoder()

y = LE.fit_transform(label)
num_classes = np.unique(y)
print(num_classes)

# Word Tokenization 
max_words = 5000  
max_len = 200    

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(feature)
sequences = tokenizer.texts_to_sequences(feature)

X = pad_sequences(sequences, maxlen=max_len)
print(tokenizer.texts_to_sequences("i am not a great guy"))

#  Preprocessing the validation dataset
val_feature  = lower_case_and_remove_punctuation(val_feature)
Sequences2 = tokenizer.texts_to_sequences(val_feature)

val_X = pad_sequences(Sequences2, maxlen=max_len)

val_y = LE.fit_transform(val_label)

#  Model Architecture serving Accuracy = 94.5 and Validation Accuracy = 94.10
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(len(num_classes), activation='softmax'))
model.summary()

# Optimizers and Loss Function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#  Model Training 
history = model.fit(X, y,batch_size = 76, epochs=10, validation_data=(val_X, val_y))

#  Accuracy and Validation Accuracy Visualization
import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()

# Simple Use Cases
sample2 = [
        "do you know about that circus ?",
        "This admin is really good",
        "I just hate this Twitter",
        "Hmmmmmmmmmmm"
          ]
example2 = tokenizer.texts_to_sequences(sample2)

new_texts2 = pad_sequences(example2, maxlen=max_len)

prediction2 = model.predict(new_texts2)

predicted_indices = np.argmax(prediction2, axis=1)
predicted_labels = LE.inverse_transform(predicted_indices)

for sample, label in zip(sample2, predicted_labels):
    print(f"Sample: {sample} -> Predicted label: {label}")