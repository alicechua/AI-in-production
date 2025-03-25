import onnxruntime
import numpy as np
import torch
from transformers import BertTokenizer

# Load the ONNX model
onnx_model_path = "app/models/classification_model_bert-base-uncased.onnx" # downloaded model from hugging face
session = onnxruntime.InferenceSession(onnx_model_path)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def classify_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='np', padding=True, truncation=True, max_length=512)

    # Extract the input_ids and attention_mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # BERT typically uses token_type_ids for sentence pairs, but for single sentence classification, you can set them to 0s
    token_type_ids = np.zeros_like(input_ids)  # Create a zero array for token_type_ids
    
    # Prepare inputs for ONNX runtime
    ort_inputs = {
        'input_ids': input_ids.astype(np.int64),
        'attention_mask': attention_mask.astype(np.int64),
        'token_type_ids': token_type_ids.astype(np.int64)  # Include token_type_ids here
    }

    # Run inference
    outputs = session.run(None, ort_inputs)
    
    # Extract logits (the raw output before softmax)
    logits = outputs[0]  # Logits will have shape (batch_size, sequence_length, num_classes)
    
    # Logits shape should be (1, seq_len, num_classes)
    print(f"Logits shape: {logits.shape}")
    
    # Extract the logits of the [CLS] token (the first token in the sequence)
    cls_token_logits = logits[:, 0, :]  # (1, num_classes), because [CLS] token is at position 0
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(torch.tensor(cls_token_logits), dim=-1)
    
    # Get the predicted class (index of the highest probability)
    predicted_class = torch.argmax(probs, dim=-1).item()  # Get index of highest probability for the class
    
    return predicted_class, probs

# Test the classification
text = "Your input text goes here."
predicted_class, probs = classify_text(text)

# Print the results
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {probs}")