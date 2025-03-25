import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load pre-trained distilBART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Example text
text = "Your long document goes here. This is an example of how you can use distilBART for summarization."

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

# Set the model to evaluation mode
model.eval()

# Specify the ONNX file path
onnx_model_path = "summarization_model_distilbart-cnn-12-6.onnx"

# Export the model to ONNX format
torch.onnx.export(model,               # The model to convert
                  (inputs["input_ids"], inputs["attention_mask"]),  # Inputs with attention_mask
                  onnx_model_path,         # Output ONNX model path
                  input_names=["input_ids", "attention_mask"],  # Input names
                  output_names=["logits"],  # Output names
                  dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_length"},  # Variable batch size and sequence length
                                "attention_mask": {0: "batch_size", 1: "seq_length"},
                                "logits": {0: "batch_size", 1: "seq_length"}},  # Variable batch size and sequence length
                  opset_version=14)  # Set ONNX opset version

print("ONNX model saved at", onnx_model_path)