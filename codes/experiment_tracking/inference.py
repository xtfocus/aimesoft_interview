"""
Inference utils
"""

import numpy as np
import torch


def get_predictions(model, data_loader, device):
    """
    Get predictions on data_loader

    Parameters
    ----------
        model: BERT model
        data_loader: Torch dataloader
        device: cpu or cuda

    Return
    ------
        Tuple of texts, prediction (binary), probability, actual target values
    """

    # Set the model to evaluation mode
    model.eval()

    # Lists to store review texts, predictions, prediction probabilities, and true labels
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    # Iterate through the data_loader
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        texts = d["text"]  # Assuming your data loader includes review texts

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get predicted labels (whichever class of higher probability) and predicted probabilities
        _, preds = torch.max(outputs, dim=1)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        review_texts.extend(texts)
        predictions.extend(preds.cpu().numpy())
        prediction_probs.extend(probs.cpu().numpy())
        real_values.extend(labels.cpu().numpy())

    # Convert the lists to numpy arrays
    review_texts = np.array(review_texts)
    predictions = np.array(predictions)
    prediction_probs = np.array(prediction_probs)
    real_values = np.array(real_values)

    accuracy = np.sum(predictions == real_values) / len(real_values)
    print(f"Accuracy: {accuracy:.4f}")

    return review_texts, predictions, prediction_probs, real_values
