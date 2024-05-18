import tensorflow as tf
from tensorflow.keras import layers, models

# Define CNN model for feature extraction
def cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    return model

# Define RNN model for sequence modeling
def rnn_model():
    model = models.Sequential()
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128))
    return model

# Combine CNN and RNN
def hybrid_model(cnn_model, rnn_model, num_classes):
    cnn_output_shape = cnn_model.output_shape[1:]
    
    cnn_input = layers.Input(shape=cnn_output_shape)
    rnn_input = layers.Input(shape=(None, cnn_output_shape[-1]))  # None for variable sequence length
    
    # Use TimeDistributed layer to apply RNN to each time step
    rnn_output = layers.TimeDistributed(rnn_model)(rnn_input)
    
    # GlobalMaxPooling1D to aggregate sequence information
    pooled_output = layers.GlobalMaxPooling1D()(rnn_output)
    
    # Combine CNN and RNN outputs
    combined = layers.concatenate([cnn_input, pooled_output])
    
    # Add output layer
    output = layers.Dense(num_classes, activation='softmax')(combined)
    
    model = models.Model(inputs=[cnn_input, rnn_input], outputs=output)
    return model

# Define input shapes and number of classes
input_shape = (64, 64, 1)  # Adjust according to your image size
num_classes = 24  # Since you have 24 classes

# Create CNN and RNN models
cnn = cnn_model(input_shape)
rnn = rnn_model()

# Combine both models
hybrid = hybrid_model(cnn, rnn, num_classes)

# Compile the model
hybrid.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
hybrid.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data generators
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'Data',
        target_size=(64, 64),  # Adjust based on your image size
        batch_size=32,
        color_mode='grayscale',  # Convert images to grayscale
        class_mode='categorical',
        shuffle=True)

# Define input shapes and number of classes
input_shape = (64, 64, 1)  # Adjust according to your image size
num_classes = 24  # Since you have 24 classes

# Create CNN and RNN models
cnn = cnn_model(input_shape)
rnn = rnn_model()

# Combine both models
hybrid = hybrid_model(cnn, rnn, num_classes)

# Compile the model
hybrid.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
hybrid.summary()

# Train the model
history = hybrid.fit(train_generator, epochs=10)  # Adjust number of epochs as needed
