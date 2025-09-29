import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import urllib.request
from DataTransformation import DataTransformation
from ClassifyOldTestament import GPTModel, evaluate_model
import polars as pl
import time

torch.manual_seed(100)

# this class takes a polars dataframe and makes a torch dataset
# so this data is pre-filtered on editors
class EditorDataset(Dataset):
    def __init__(self, polars_df, labels_column="data_set", max_length=None, pad_token_id=22702):
        self.polars_df = polars_df
        self.labels_column = labels_column
        # get list of lists
        self.tokens_list = [
            x for x in self.polars_df["Token"].to_list()
        ]
        #
        # pad sequences to max_length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        # oh, so 'line' is really a single token, and
        # from the original example it is...a line?
        self.tokens = [
            line + [pad_token_id] * (max_length - len(line))
            for line in self.tokens_list
        ]

    def __getitem__(self, index):
        encoded = self.tokens[index]
        label = self.polars_df.select(pl.col(self.labels_column)).row(index)[0]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.tokens_list)

    def _longest_encoded_length(self):
        return max(len(line) for line in self.tokens_list)

def ft_calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += ( (predicted_labels == target_batch).sum().item() )

        else:
            break
    return correct_predictions / num_examples

def ft_calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = ft_calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def ft_calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    assert logits.shape[0] == target_batch.shape[
        0], f"Logits batch {logits.shape[0]} != target batch {target_batch.shape[0]}"
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def ft_evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = ft_calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = ft_calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = ft_calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = ft_evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss: {train_loss:.3f}, "
                      f"Val loss: {val_loss:.3f}"
                      )

        train_accuracy = ft_calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = ft_calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training Accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def plot_values(
        epochs_seen, examples_seen, train_values, val_values, labels="loss"
):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {labels}")
    ax1.plot(
        epochs_seen, train_values, label=f"Training {labels}"
    )
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {labels}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(labels.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{labels}-plot.pdf")

if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 22703,  # this is the vocab size of the hebrew bible text
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 8,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # reference material
    passages = {
        "D": ["Deut 6", "Deut 12-13", "Deut 15-16", "Deut 18-19", "Deut 26", "Deut 28"],
        "DH": ["Deut 8-11", "Deut 27", "Josh 1", "Josh 5", "Josh 6", "Josh 12", "Josh 23",
               "Judg 2", "Judg 6", "2Sam 7", "1Kgs 8", "2Kgs 17:1-21", "2Kgs 22-25"],
        "P": ["Gen 1:1-31", "Gen 2:1-3", "Gen 5:3-28", "Gen 5:30-32", "Gen 6:9-22", "Gen 9:1-17",
              "Gen 6:28-29", "Gen 10:2-7", "Gen 10:20", "Gen 10:22-23", "Gen 10:31", "Gen 11:11-26",
              "Gen 11:29-32", "Gen 12:5", "Gen 13:6", "Gen 13:12", "Gen 16:3", "Gen 16:15-16", "Gen 21:2-5",
              "Gen 22:20-24", "Gen 23:1-20", "Gen 25:7-10", "Gen 25:13-17", "Gen 25:20", "Gen 26:20",
              "Gen 26:34-35", "Gen 27:46", "Gen 28:1-9", "Gen 35:9-15", "Gen 35:27-29", "Gen 36:40-43",
              "Gen 37:1", "Gen 46:6-7", "Gen 47:28", "Gen 49:29-33", "Gen 50:12-13", "Exod 1:1-4", "Exod 1:7",
              "Exod 1:13-14", "Exod 2:23-25", "Exod 7:1-13", "Exod 7:19-22", "Exod 8:1-3", "Exod 8:11-15",
              "Exod 9:8-12", "Exod 11:9-10", "Exod 12:40-42", "Exod 13:20", "Exod 14:1-4", "Exod 14:8-10",
              "Exod 14:15-18", "Exod 14:21-23", "Exod 14:27-29", "Exod 15:22", "Exod 19:1", "Exod 24:16-17",
              "Gen 17", "Exod 6", "Exod 16", "Exod 25-31", "Exod 35-40", "Lev 1-4", "Exod 8-9"]
    }

    # # Step through the procedure
    dt = DataTransformation(file_name='wlc.txt', passages=passages)
    dt.initial_transform()
    dt.assign_editors()
    dt.add_training_testing()

    # create training data polars dataframe
    train_data = dt.df.filter(
        pl.col("data_set") == "train"
    ).select(["Token","label"])

    # create validation data polars dataframe
    val_data = dt.df.filter(
        pl.col("data_set") == "val"
    ).select(["Token","label"])

    # create test data polars dataframe
    test_data = dt.df.filter(
        pl.col("data_set") == "test"
    ).select(["Token","label"])

    # Create datasets
    train_dataset = EditorDataset(train_data, labels_column="label", max_length=50)
    val_dataset = EditorDataset(val_data, labels_column="label", max_length=50)
    test_dataset = EditorDataset(test_data, labels_column="label", max_length=50)

    # need to load the model from model.pth
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # modify the model for classification

    num_classes = 3
    model.out_head = torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"],
        out_features=num_classes
    )

    # make the last layer trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # instantiate data loaders
    num_workers = 0
    batch_size = 8
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    # run the training loop
    start_time = time.time()
    torch.manual_seed(100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 3

    device = torch.device("cpu")

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )
    end_time = time.time()
    execution_time_minutes = ( end_time - start_time ) / 60
    print(f"Training completed in {execution_time_minutes: .2f} minutes.")

    # plot some plots
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, labels="accuracy")

    # overall accuracy
    train_accuracy = ft_calc_accuracy_loader(train_loader, model, device)
    val_accuracy = ft_calc_accuracy_loader(val_loader, model, device)
    test_accuracy = ft_calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # now save the model to be used later
    torch.save(model.state_dict(), "review_classifer.pth")

