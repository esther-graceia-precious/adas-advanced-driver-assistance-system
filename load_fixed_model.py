from model_arch import build_model

model = build_model()
model.load_weights("driver_distraction_model.h5")

# Save in modern format
model.save("driver_model.keras")

print("✅ Model converted successfully")
