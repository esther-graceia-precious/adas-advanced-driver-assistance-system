import tensorflow as tf
from preprocessing import train_data, val_data
from model import model

EPOCHS = 6   # enough for review-1

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
model.save("driver_distraction_model.h5")
