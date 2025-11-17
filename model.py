import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models, callbacks, Input
import matplotlib.pyplot as plt

# --- Paths ---
train_dir = "data/train"
val_dir = "data/val"
img_size = (224, 224)
batch_size = 32

# --- Data Generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

# --- Build Model (EfficientNetV2B0 with RGB input) ---
inputs = Input(shape=(224, 224, 3))
base_model = EfficientNetV2B0(weights="imagenet", include_top=False, input_tensor=inputs)
base_model.trainable = False  # freeze initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = callbacks.ModelCheckpoint("best_model_rgb.h5", save_best_only=True, monitor="val_accuracy", mode="max")
earlystop_cb = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# --- Train (Feature Extraction) ---
history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[checkpoint_cb, earlystop_cb])

# --- Fine-tuning ---
base_model.trainable = True
for layer in base_model.layers[:-20]:  # unfreeze last 20 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[checkpoint_cb, earlystop_cb])

model.save("skin_cancer_model_rgb_v2.h5")
print("✅ Model training complete. Saved as skin_cancer_model_rgb_v2.h5")

# --- Plot Accuracy & Loss ---
acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
loss = history.history['loss'] + fine_tune_history.history['loss']
val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()
