import tensorflow as tf
from preprocessing import train_data, val_data
from model import model

EPOCHS = 6   

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy Curve")
plt.show()


# Save model
model.save("driver_model.keras")
