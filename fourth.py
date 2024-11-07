import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer

# Define small corpus
corpus = ["the quick brown fox jumped over the lazy dog"]

# Tokenize the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # Adding 1 for padding token
window_size = 2  # Context window size

# Prepare training data (context, target word pairs)
data, labels = [], []
for sentence in tokenizer.texts_to_sequences(corpus):
    for i in range(window_size, len(sentence) - window_size):
        context = [sentence[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = sentence[i]
        data.append(context)
        labels.append(target)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Define and train the model
model = models.Sequential([
    layers.Embedding(input_dim=total_words, output_dim=50, input_length=window_size * 2),
    layers.GlobalAveragePooling1D(),
    layers.Dense(total_words, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=50, verbose=1)

# Get word embeddings and find most similar words to a target
word_embeddings = model.layers[0].get_weights()[0]
target_word = 'quick'  # Example target word

target_idx = tokenizer.word_index.get(target_word)
if target_idx:
    target_embedding = word_embeddings[target_idx - 1]
    similarities = np.dot(word_embeddings, target_embedding) / (np.linalg.norm(word_embeddings, axis=1) * np.linalg.norm(target_embedding))
    similar_idx = similarities.argsort()[-5:][::-1]
    similar_words = [word for word, idx in tokenizer.word_index.items() if idx in similar_idx]
    print(f"Most similar words to '{target_word}': {similar_words}")
