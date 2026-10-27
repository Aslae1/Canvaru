"""
Train the CNN used by CANVARU.

Dataset: Google Quick, Draw! bitmap files.

The script downloads six public QuickDraw .npy class files,
converts them to 28x28 grayscale images, and trains a compact CNN.

Run:
    python train_doodle_model.py

The resulting model is saved to:
    models/doodle_cnn.keras
"""

import os
import urllib.request

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. Configuration
# ============================================================

CLASSES = ["apple", "cat", "flower", "house", "star", "tree"]

BASE = (
    "https://storage.googleapis.com/quickdraw_dataset/"
    "full/numpy_bitmap/"
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

X_parts = []
y_parts = []

per_class = 3500


# ============================================================
# 2. Load QuickDraw Dataset
# ============================================================

for label, name in enumerate(CLASSES):

    path = os.path.join("data", f"{name}.npy")

    if not os.path.exists(path):
        print("Downloading", name)

        urllib.request.urlretrieve(
            BASE + name.replace(" ", "%20") + ".npy",
            path
        )

    print("Loading", name)

    data = np.load(path)

    # Use 3500 samples from each class
    data = data[:per_class].astype("float32") / 255.0

    # Convert to 28x28 grayscale images
    X_parts.append(
        data.reshape(-1, 28, 28, 1)
    )

    # Assign numerical label
    y_parts.append(
        np.full(len(data), label)
    )


# ============================================================
# 3. Combine Dataset
# ============================================================

X = np.concatenate(X_parts)
y = np.concatenate(y_parts)

print()
print("Total samples:", len(X))
print("Image shape:", X.shape)


# ============================================================
# 4. Shuffle Dataset
# ============================================================

rng = np.random.default_rng(42)

idx = rng.permutation(len(X))

X = X[idx]
y = y[idx]


# ============================================================
# 5. Train-Test Split
# ============================================================

split = int(len(X) * 0.85)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. Build CNN Model
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First convolution block
    tf.keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Convert feature maps into vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.25),

    # Six output classes
    tf.keras.layers.Dense(
        len(CLASSES),
        activation="softmax"
    )
])


# ============================================================
# 7. Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. Display Model Architecture
# ============================================================

model.summary()


# ============================================================
# 9. Train CNN
# ============================================================

print()
print("Starting CNN training...")
print()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=8,
    batch_size=128
)


# ============================================================
# 10. Accuracy Curve - Figure 5.1
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 9),
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    range(1, 9),
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.xticks(range(1, 9))

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. Loss Curve - Figure 5.2
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 9),
    history.history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, 9),
    history.history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.xticks(range(1, 9))

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. Evaluate CNN on Test Dataset
# ============================================================

print()
print("Evaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# 13. Generate Predictions
# ============================================================

print()
print("Generating predictions...")

y_pred_probability = model.predict(
    X_test,
    verbose=0
)

# Select class with highest probability
y_pred = np.argmax(
    y_pred_probability,
    axis=1
)


# ============================================================
# 14. Confusion Matrix - Figure 5.3
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print()
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "Confusion Matrix of the CNN"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. Save Trained Model
# ============================================================

model.save(
    "models/doodle_cnn.keras"
)

print()
print("===================================")
print("Training completed successfully.")
print("===================================")

print()
print("Generated files:")

print("1. training_validation_accuracy.png")
print("2. training_validation_loss.png")
print("3. confusion_matrix.png")
print("4. models/doodle_cnn.keras")
from sklearn.metrics import classification_report

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASSES,
        digits=4
    )
)