
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Add, Activation,BatchNormalization
from keras.models import Model


# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
# emotion_model = Sequential()

# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax'))





input_shape = (48, 48, 1)  # Update with the actual input shape
num_classes = 7  # Replace with the actual number of classes

# Define the model
input_layer = Input(shape=input_shape)

# Convolutional Layer 1
conv1 = Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(input_layer)

# MaxPooling Layer 1
maxpool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv1)

# Convolutional Layer 2
conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(maxpool1)

# Residual Block 1
conv1_res1 = Conv2D(64, (1, 1), padding='same', activation='relu')(conv2)
conv2_res1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1_res1)
conv3_res1 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2_res1)
conv4_res1 = Conv2D(256, (1, 1), padding='same')(conv3_res1)

# Skip connection
skip_connection_res1 = Conv2D(256, (1, 1), padding='same')(conv2)
added_res1 = Add()([conv4_res1, skip_connection_res1])
res1_output = Activation('relu')(added_res1)

# Convolutional Layer 3
conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(res1_output)

# MaxPooling Layer 2
maxpool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv3)

# Convolutional Layer 4
conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(maxpool2)

# Residual Block 2
conv1_res2 = Conv2D(64, (1, 1), padding='same', activation='relu')(conv4)
conv2_res2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1_res2)
conv3_res2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2_res2)
conv4_res2 = Conv2D(256, (1, 1), padding='same')(conv3_res2)

# Skip connection
skip_connection_res2 = Conv2D(256, (1, 1), padding='same')(conv4)
added_res2 = Add()([conv4_res2, skip_connection_res2])
res2_output = Activation('relu')(added_res2)

# Convolutional Layer 5
conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(res2_output)

# MaxPooling Layer 3
maxpool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv5)

# Convolutional Layer 6
conv6 = Conv2D(512, (3, 3), padding='same', activation='relu')(maxpool3)

# Fully Connected Layer 1 (FC1)
flatten = Flatten()(conv6)
fc1 = Dense(1024, activation='relu')(flatten)
fc1 = BatchNormalization()(fc1)
fc1 = Dropout(0.5)(fc1)  # Adjust dropout rate as needed

# Fully Connected Layer 2 (FC2)
fc2 = Dense(512, activation='relu')(fc1)
fc2 = BatchNormalization()(fc2)
fc2 = Dropout(0.5)(fc2)  # Adjust dropout rate as needed

# Output layer
output_layer = Dense(num_classes, activation='softmax')(fc2)  # Replace num_classes with the actual number of classes

# Create the model

model = Model(inputs=input_layer, outputs=output_layer)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




cv2.ocl.setUseOpenCL(False)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the neural network/model
emotion_model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in jason file
model_json = model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
model.save_weights('emotion_model.h5')

