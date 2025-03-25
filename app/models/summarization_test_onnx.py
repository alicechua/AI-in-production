import onnxruntime as ort
from transformers import BartTokenizer
import numpy as np

# Load the tokenizer for distilBART
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Example long document to summarize
text = """"
In recent years, renewable energy has gained considerable attention worldwide as the urgency of combating climate change has become more pronounced. Governments, corporations, and environmental organizations are increasingly pushing for a shift away from fossil fuels toward cleaner, more sustainable energy sources. This transition to renewable energy, which includes solar, wind, hydroelectric, and geothermal power, has profound implications not only for the environment but also for global economies.

The global demand for renewable energy has surged in response to the growing need for sustainability and the depletion of nonrenewable resources. Solar energy, for example, has seen significant advances in technology, making it a cost-effective and viable option for both residential and industrial use. Wind energy is another rapidly growing sector, with large-scale wind farms being built in countries like the United States, China, and several European nations. These investments in renewable energy infrastructure have led to the creation of millions of jobs in the clean energy sector, helping to bolster economies in many regions.

However, the transition to renewable energy is not without its challenges. One of the primary obstacles is the need for significant investment in infrastructure, particularly in countries that have historically relied on coal, oil, or natural gas for energy production. In addition, while renewable energy sources are abundant, they are also intermittent—solar power is only available during the day, and wind energy is dependent on weather conditions. This intermittency creates challenges in maintaining a stable and reliable energy grid, which requires advanced energy storage solutions and smart grid technologies to balance supply and demand.

Another issue that arises with the shift toward renewable energy is the economic impact on communities that rely heavily on the fossil fuel industry. Many coal mining towns, for example, face job losses and economic hardship as the demand for coal declines. Governments are tasked with finding ways to support these communities, either by transitioning workers to jobs in the renewable energy sector or providing retraining programs to help them develop new skills for the modern economy.

Despite these challenges, the transition to renewable energy is seen as essential for reducing greenhouse gas emissions and mitigating the effects of climate change. The Paris Agreement, signed by nearly 200 countries, aims to limit global warming to well below 2°C, with an aspiration of limiting it to 1.5°C. Achieving these goals requires a massive global effort to cut carbon emissions, and the energy sector is at the heart of this effort.

In conclusion, the rise of renewable energy presents both opportunities and challenges for global economies. While the transition to cleaner energy sources can drive economic growth and reduce environmental damage, it also requires significant investment, infrastructure development, and social adaptation. The next few decades will be crucial in determining how well the world can balance these competing priorities and whether renewable energy can play a key role in the fight against climate change.
"""

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)  # Ensure length is appropriate

# Load the ONNX model
onnx_session = ort.InferenceSession("summarization_model_distilbart-cnn-12-6.onnx")  # Replace with your actual ONNX model path

# Prepare the inputs for ONNX inference (convert tensors to NumPy arrays)
onnx_inputs = {
    "input_ids": inputs["input_ids"].numpy(),
    "attention_mask": inputs["attention_mask"].numpy()  # Include attention_mask as well
}

# Run inference
onnx_outputs = onnx_session.run(None, onnx_inputs)

# Get the output logits (usually logits are the first output)
logits = onnx_outputs[0]

# Apply softmax to logits to get probabilities, then get the most probable token IDs
probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
generated_ids = np.argmax(probabilities, axis=-1)

# Flatten the generated_ids to ensure it's a 1D array
generated_ids = np.array(generated_ids).flatten()

# Decode the tokens to generate the summary
summary = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"Generated Summary: {summary}")
