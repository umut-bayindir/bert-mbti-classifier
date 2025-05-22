import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from data_loader import MBTIDataset
from model import MBTIClassifier


def train(
    questions_file, responses_file, output_dir,
    epochs=3, batch_size=8, lr=2e-5, max_length=128,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    os.makedirs(output_dir, exist_ok=True)
    dataset = MBTIDataset(questions_file, responses_file, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MBTIClassifier(num_labels=4).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        # save checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch{epoch+1}.pt"))

    print("Training complete.")