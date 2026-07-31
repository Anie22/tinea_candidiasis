import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_DIR = "data"
MODEL_DIR = "model"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 6

CONFIDENCE_THRESHOLD = 0.75
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# DATASET
# ==========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

class_names = train_ds.class_names

print("Detected Classes:", class_names)

# ==========================================================
# AUGMENTATION
# ==========================================================

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])

def preprocess(images, labels, training=False):
    if training:
        images = augment(images, training=True)

    images = preprocess_input(images)

    return images, labels


val_ds_raw = val_ds

train_ds = (
    train_ds
    .map(lambda x, y: preprocess(x, y, True))
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .map(lambda x, y: preprocess(x, y, False))
    .prefetch(tf.data.AUTOTUNE)
)

# ==========================================================
# MODEL
# ==========================================================

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ==========================================================
# CALLBACKS
# ==========================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "tinea_candidiasis_skin_classifier.keras"),
    monitor="val_accuracy",
    save_best_only=True,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1,
)

callbacks = [
    checkpoint,
    early_stop,
    reduce_lr,
]

# ==========================================================
# STAGE 1
# ==========================================================

print("\nTraining classifier head...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
)

# ==========================================================
# STAGE 2
# ==========================================================

print("\nFine tuning MobileNetV2...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks,
)

# ==========================================================
# SAVE LABELS
# ==========================================================

with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
    json.dump(
        {
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        f,
        indent=2,
    )

# ==========================================================
# TRAINING HISTORY
# ==========================================================

acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(acc,label="Train")
plt.plot(val_acc,label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label="Train")
plt.plot(val_loss,label="Validation")
plt.title("Loss")
plt.legend()

plt.tight_layout()

# plt.savefig(
#     os.path.join(MODEL_DIR,"training_history.png"),
#     dpi=150
# )

plt.close()

# ==========================================================
# EVALUATION
# ==========================================================

y_true = []
y_pred = []

for images, labels in val_ds_raw:

    predictions = model.predict(
        preprocess_input(
            tf.cast(images, tf.float32)
        ),
        verbose=0,
    )

    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(range(len(class_names)), class_names)

plt.yticks(range(len(class_names)), class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

# plt.savefig(
#     os.path.join(MODEL_DIR, "confusion_matrix.png"),
#     dpi=150,
# )

plt.close()

report = classification_report(y_true, y_pred, target_names=class_names)

print(report)

# with open(
#     os.path.join(MODEL_DIR, "classification_report.txt"),
#     "w",
# ) as f:
#     f.write(report)

print("\nTraining Complete!")