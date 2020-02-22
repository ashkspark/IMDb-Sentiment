import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)

print(train_data[0])

#use imdb dictionary to map numbers to words, see tensorflow 2.0 tutorial

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def translate_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print(translate_review(train_data[1]))

#Are the reviews of the same lenght?
s = bool(len(test_data[1]) == len(test_data[0]))
if s:
    print("Yes")
else:
    print("No")
#####################################
#We need to make the reviews to be of the same lenght for neural nets

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)

#Are the reviews of the same lenght?
s = bool(len(test_data[1]) == len(test_data[0]))
if s:
    print("Yes")
else:
    print("No")
#####################################

# Creating a neural net model

model = keras.Sequential()
model.add(keras.layers.Embedding(50000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))


model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000] #validation data
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1) #batch_size to prevent out of memory let's say
results = model.evaluate(test_data, test_labels)
print(results) #print accuracy (loss, accuracy)




####################################
model.save("IMDb_Sentiment_Compiler.h5")
#####################################






#Let's visualize the prediction
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(translate_review(test_review))
print("Prediction Sentiment: " + str(predict[0]))
print("Actual Sentiment: " + str(test_labels[0]))
####################################
#Let's examine how the predictive model performs for a review of the Lion King movie
####################################
pred_model = keras.models.load_model("IMDb_Sentiment_Compiler.h5")


#######################################
def translate_review_D(s):
     encoded = [1]
     for word in s:
         if word.lower() in word_index:
              encoded.append(word_index[word.lower()])
     else:
         encoded.append(2)

     return encoded
########################################


########################################################################
with open("lion_king_review.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        translation = translate_review_D(nline)
        translation = keras.preprocessing.sequence.pad_sequences([translation], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)
        print(translation[0])
        prediction = pred_model.predict(translation[0])
        print(line)
        #print(translation)
        print(prediction[0])
########################################################################








