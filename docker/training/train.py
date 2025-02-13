import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score

def main():
    # Example hyperparameters. In SageMaker, you could also parse environment variables or JSON for real usage.
    epochs = 2
    batch_size = 4
    learning_rate = 0.001

    # Typical directories for SageMaker.
    train_data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_data_dir = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # Simple transforms for images
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    # Load training and validation sets from local directories
    train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(val_data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # A simple binary classifier (e.g., small CNN or pretrained backbone). 
    # For demonstration, we'll do a small sequential model:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224*224*3, 128),
        nn.ReLU(),
        nn.Linear(128, 2)  # two classes: tennis court vs. not tennis court
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} complete.")

    # Validation / metrics
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate accuracy and recall for two-class problem (pos_label=1).
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="binary", pos_label=1)

    print("Validation Accuracy:", accuracy)
    print("Validation Recall:", recall)

    # Save metrics to /opt/ml/output/metrics.json so that the GitHub Action can fetch them
    metrics = {
        "accuracy": accuracy,
        "recall": recall
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")
    print(metrics_path, os.listdir(metrics_path))

    # Save the model (PyTorch save example)
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 