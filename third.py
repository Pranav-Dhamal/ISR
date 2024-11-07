import tensorflow as tf
from tensorflow.keras import layers, models


(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))


encoder_input = layers.Input(shape=(784,))
encoded = layers.Dense(32, activation='relu')(encoder_input)


decoded = layers.Dense(784, activation='sigmoid')(encoded)


autoencoder = models.Model(encoder_input, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, shuffle=True)


reconstructed = autoencoder.predict(x_test)


reconstructed = autoencoder.predict(x_test)


import matplotlib.pyplot as plt

n = 5  # Number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")
plt.show()
