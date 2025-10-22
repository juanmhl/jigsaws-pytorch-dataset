import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

from jigsaws_pytorch_dataset import KinematicsDataset
from jigsaws_pytorch_dataset.options import (
    KinematicsSamplingMode,
    LabelsFormat,
    Users,
    UnlabeledDataPolicy,
)
from jigsaws_pytorch_dataset.transforms import extract_PSM_kinematics
from jigsaws_pytorch_dataset.data_scalers.scalers import MinMaxScaler
from jigsaws_pytorch_dataset.collate_fns import collate_fn_seqs_with_padding


# --- 1. Define the LSTM Model ---
class GestureLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5
    ):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(
            hidden_size * 2, num_classes
        )  # Multiply by 2 for bidirectional

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Forward pass through LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout
        output = self.dropout(output)

        # Pass through linear layer
        logits = self.fc(output)
        return logits


def main():
    # --- 2. Setup Hyperparameters and Device ---
    INPUT_SIZE = 24  # Number of features from extract_PSM_kinematics
    HIDDEN_SIZE = 384
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DATA_DIR = "dataset/Suturing"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TensorBoard Setup ---
    writer = SummaryWriter("runs/lstm_experiment_1")

    # --- 3. Prepare Datasets and DataLoaders ---
    train_users = (Users.B, Users.C, Users.D, Users.E, Users.F)
    val_users = (Users.G, Users.H, Users.I)

    # Create training dataset
    train_dataset = KinematicsDataset(
        dir=DATA_DIR,
        mode=KinematicsSamplingMode.SEQUENCE,
        users_set=train_users,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        transform=extract_PSM_kinematics,
    )

    # Create validation dataset
    val_dataset = KinematicsDataset(
        dir=DATA_DIR,
        mode=KinematicsSamplingMode.SEQUENCE,
        users_set=val_users,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        transform=extract_PSM_kinematics,
    )

    # Fit scaler on training data and apply to both
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_dataset.fit_scaler(scaler)
    val_dataset.set_scaler(scaler)

    # Get number of classes from the dataset's gesture map
    NUM_CLASSES = len(train_dataset.gesture_map)
    print(f"Number of classes: {NUM_CLASSES}")

    # Fix random seed for reproducibility
    torch.manual_seed(42)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_seqs_with_padding,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_seqs_with_padding,
    )

    # --- 4. Initialize Model, Loss, and Optimizer ---
    model = GestureLSTM(
        INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE
    ).to(device)
    criterion = nn.CrossEntropyLoss()  # Handles softmax internally
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # --- 5. Training and Validation Loop ---
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        for features, labels, lengths in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features, lengths)

            # Loss calculation (needs reshaping)
            # outputs: (batch, max_len, num_classes) -> (batch * max_len, num_classes)
            # labels: (batch, max_len) -> (batch * max_len)
            loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features, lengths)
                loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
                total_val_loss += loss.item()

                # Accuracy calculation (ignore padding)
                _, predicted = torch.max(outputs, 2)
                for i in range(len(lengths)):
                    seq_len = lengths[i]
                    correct_predictions += (
                        (predicted[i, :seq_len] == labels[i, :seq_len]).sum().item()
                    )
                    total_samples += seq_len

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = (
            (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%"
        )

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

    writer.close()


if __name__ == "__main__":
    main()

# --- How to use TensorBoard with VSCode Remote Tunnels ---
#
# 1. Run this script on your remote machine.
#    This will create a `runs/lstm_experiment_1` directory containing the training logs.
#
# 2. Open a new terminal in VSCode (Terminal > New Terminal).
#
# 3. In the new terminal, start TensorBoard by pointing it to the log directory:
#    tensorboard --logdir=runs
#
# 4. TensorBoard will start a web server, typically on port 6006.
#    VSCode's remote extension should automatically detect this and show a notification
#    asking if you want to forward the port. Click "Open in Browser".
#
# 5. If you don't see a notification, you can manually forward the port:
#    - Go to the "Ports" tab in the bottom panel of VSCode.
#    - Click "Forward a Port" and enter `6006`.
#    - Open `http://localhost:6006` in your local web browser.
#
# 6. You can now view the training graphs and metrics in the TensorBoard dashboard.
