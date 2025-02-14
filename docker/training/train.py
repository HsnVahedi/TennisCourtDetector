import os
import json
import torch
import mlflow
import mlflow.pytorch   # for pytorch model logging
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score

def main():
    # Example hyperparameters might be read from environment or script arguments
    epochs = int(os.environ.get("epochs", 2))
    batch_size = int(os.environ.get("batch_size", 4))
    learning_rate = 0.001

    # ----------------------------------------------------------------------------
    # ADDED: MLflow tracking URI from environment (provided by GitHub actions or the Docker environment).
    # If 'MLFLOW_TRACKING_ARN' is set in the environment, we can pass it to mlflow.set_tracking_uri(...).
    # ----------------------------------------------------------------------------
    tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN", "")
    if tracking_arn:
        mlflow.set_tracking_uri(tracking_arn)
    
    # ----------------------------------------------------------------------------
    # ADDED: Turn on MLflow autologging for PyTorch
    # ----------------------------------------------------------------------------
    mlflow.pytorch.autolog()

    train_data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_data_dir   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    output_dir     = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
    model_dir      = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
    val_dataset   = torchvision.datasets.ImageFolder(val_data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224*224*3, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    recall   = recall_score(all_labels, all_preds, average="binary", pos_label=1)
    print("Validation Accuracy:", accuracy)
    print("Validation Recall:", recall)

    # Save metrics to the model dir for reference, though MLflow autologging also captures them.
    metrics = {"accuracy": accuracy, "recall": recall}
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

    # Save the model
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ----------------------------------------------------------------------------
    # OPTIONAL: If you want to register the model in MLflow's registry for best practices:
    # ----------------------------------------------------------------------------
    registered_model_name = os.environ.get("MLFLOW_MODEL_NAME", "TennisCourtDetectionModel")
    with mlflow.start_run():   
        # log_model can automatically register if you specify 'registered_model_name'
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        mlflow.log_metrics({"accuracy": accuracy, "recall": recall})

if __name__ == "__main__":
    main() 