import matplotlib.pyplot as plt

train_acc = [
0.7860,0.8603,0.8820,0.8895,0.8919,0.9063,0.8971,0.9091,0.9124,0.9184,
0.9225,0.8991,0.9150,0.9245,0.9229,0.9273,0.9231,0.9210,0.9298,0.9296
]

val_acc = [
0.8784,0.9009,0.9052,0.9169,0.9165,0.9260,0.9247,0.9083,0.9269,0.9316,
0.9394,0.9277,0.9373,0.9303,0.9433,0.9437,0.9342,0.9481,0.9507,0.9533
]

plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_graph.png")
plt.show()
