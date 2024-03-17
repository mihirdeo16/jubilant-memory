"""
This script is used to perform inference using the Hugging Face model.

To run the script, use the following command:
python hf_inference.py

Following libraries need to be installed:
pip install transformers peft torch
"""

from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
import torch

def text_preprocessing(text: str) -> str:

    # TODO: Add more preprocessing steps

    return text

if __name__ == "__main__":
    # Set the seed
    seed = 741
    model_checkpoint = "mihirdeo16/cpt-code-biomed"
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the lora model for inference
    model = AutoPeftModelForSequenceClassification.from_pretrained(model_checkpoint)

    # Set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Input text
    inputs = "Random text to test the model"

    # Preprocess the text
    text = text_preprocessing(inputs)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt",max_length=model.config.max_position_embeddings,
                            padding=True, truncation=True)

    # Get the prediction
    prediction = model(**inputs)

    # Get the result
    result_dict = {
        "label": model.config.id2label[prediction["logits"].argmax().item()],
        "score": prediction["logits"].softmax(dim=-1).max().item()
    }

    print(f"Predicted label: {result_dict['label']}, Score: {result_dict['score']}")