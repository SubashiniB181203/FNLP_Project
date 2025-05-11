from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare image augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# Load DenseNet121 and build classification model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile and train
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
                    validation_data=(X_test, y_test_cat),
                    epochs=25, callbacks=[early_stop])

# Fine-tune the model
fine_tune_at = 300
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
fine_tune_history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
                              validation_data=(X_test, y_test_cat),
                              epochs=10, callbacks=[early_stop])
