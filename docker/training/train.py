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
    epochs = int(os.environ.get("epochs", 20))
    batch_size = int(os.environ.get("batch_size", 4))
    learning_rate = 0.001

    # ----------------------------------------------------------------------------
    # ADDED: MLflow tracking URI from environment (provided by GitHub actions or the Docker environment).
    # If 'MLFLOW_TRACKING_ARN' is set in the environment, we can pass it to mlflow.set_tracking_uri(...).
    # ----------------------------------------------------------------------------
    tracking_uri = os.environ.get("MLFLOW_TRACKING_ARN")
    mlflow.set_tracking_uri(tracking_uri)
    
    # ----------------------------------------------------------------------------
    # ADDED: Turn on MLflow autologging for PyTorch
    # ----------------------------------------------------------------------------
    mlflow.pytorch.autolog()

    train_data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_data_dir   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    # output_dir     = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
    # model_dir      = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

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

    # Ensure we are *in* an MLflow run while training
    with mlflow.start_run(run_name="tennis-court-training") as run:
        # Get the run ID
        run_id = run.info.run_id
        
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

        # Log final val metrics:
        mlflow.log_metrics({"accuracy": accuracy, "recall": recall})

        # ----------------------------------------------------------------------------
        # OPTIONAL: If you want to register the model in MLflow's registry for best practices:
        # ----------------------------------------------------------------------------
        registered_model_name = os.environ.get("MLFLOW_MODEL_NAME", "TennisCourtDetectionModel")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model.tar.gz",
            registered_model_name=registered_model_name
        )

if __name__ == "__main__":
    main() 