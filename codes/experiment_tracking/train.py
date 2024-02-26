from math import ceil
from collections import defaultdict
from data_preprocessing import BATCH_SIZE
import torch
from torch import nn, optim
from transformers import AdamW, get_linear_schedule_with_warmup
from data_preprocessing import train_data_loader, test_data_loader, dev_data_loader,\
                                MODEL_NAME, class_names, train_size, dev_size
import numpy as np
import mlflow

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 7

# Function for a single training iteration
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    # Loop thru batches
    for batch_no, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        
        loss = loss_fn(outputs, labels)
        acc = torch.sum(preds == labels) / BATCH_SIZE
        # Log a batch
        step = ceil(epoch * len(data_loader) / BATCH_SIZE) + batch_no
        mlflow.log_metric(
            "Batch Train Accuracy",
            acc.item() / BATCH_SIZE,
            step=step
        )
        mlflow.log_metric(
            "Batch Train Loss",
            loss.item(),
            step=step
        )
        
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        # Backward prop
        loss.backward()
        
        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            # Get model ouptuts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(model, optimizer, scheduler, EPOCHS, device):
        
    history = defaultdict(list)
    
    for epoch in range(EPOCHS):
        
        # Show details 
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            train_size,
            epoch
        )
        
        print(f"Train loss {train_loss} accuracy {train_acc}")
        
        # Get model performance (accuracy and loss)
        val_acc, val_loss = eval_model(
            model,
            dev_data_loader,
            loss_fn,
            device,
            dev_size
        )
        
        print(f"Val   loss {val_loss} accuracy {val_acc}")
        print()
        
        
        mlflow.log_metric("Train Loss", train_loss, step=epoch)
        mlflow.log_metric("Val  Loss", val_loss, step=epoch)
        mlflow.log_metric("Train Accuracy", train_acc, step=epoch)
        mlflow.log_metric("Val Accuracy", val_acc, step=epoch)

        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
    return history
     
        
def optimizer_scheduler(model):
    
    # Optimizer Adam 
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    
    total_steps = len(train_data_loader) * EPOCHS
    
    print("len(train_data_loader)=", len(train_data_loader))
    print(f"Total training steps = {total_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler
    
# Set the loss function 
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
