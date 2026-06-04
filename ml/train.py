import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, TensorDataset
from ml.model import MLP
from ml.preprocessing import load_data, encode_df, INPUT_SIZE

DATA_PATH   = "data/adult.csv"
MODEL_PATH  = "ml/model.pth"
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
TEST_SPLIT  = 0.2
RANDOM_SEED = 42


def train():
    print("Loading data...")
    df = load_data(DATA_PATH)
    X, y = encode_df(df)
    print(f"Total rows: {len(df)}, Features: {INPUT_SIZE}")

    # Shuffle and split into train / test
    torch.manual_seed(RANDOM_SEED)
    n_total = len(X)
    n_test  = int(n_total * TEST_SPLIT)
    n_train = n_total - n_test
    perm    = torch.randperm(n_total)

    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_test,  y_test  = X[perm[n_train:]], y[perm[n_train:]]
    print(f"Train: {n_train}, Test: {n_test}")

    # Save the test split so we can run evaluation later
    torch.save({"X": X_test, "y": y_test}, "ml/test_data.pt")

    # DataLoader feeds the model in small batches instead of all at once
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    model     = MLP(INPUT_SIZE)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS}  loss: {total_loss / n_train:.4f}")

    # Evaluate on the held-out test set
    model.eval()
    with torch.no_grad():
        preds    = (model(X_test).squeeze() >= 0.5).float()
        accuracy = (preds == y_test).float().mean().item()
    print(f"\nTest accuracy: {accuracy * 100:.1f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()