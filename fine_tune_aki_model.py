import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Step 1: Define the 116 cases as a list of dictionaries
cases = [
    {"input": "65-year-old male with hypertension and diabetes due to volume depletion.", "output": "AKI likely due to volume depletion in a patient with hypertension and diabetes. Recommend IV fluids and monitor renal function."},
    {"input": "70-year-old female post-cardiac catheterization due to contrast-induced nephropathy.", "output": "AKI likely due to contrast-induced nephropathy. Recommend hydration and monitor renal function."},
    {"input": "55-year-old male with sepsis.", "output": "AKI likely due to sepsis. Recommend broad-spectrum antibiotics and fluid resuscitation."},
    # Add the remaining 113 cases here...
]

# Step 2: Convert the list of dictionaries into a pandas DataFrame
data = pd.DataFrame(cases)

# Step 3: Clean the text (if needed)
def clean_text(text):
    text = text.strip()  # Remove leading/trailing spaces
    text = text.replace("\n", " ")  # Replace newlines with spaces
    return text

data["input"] = data["input"].apply(clean_text)
data["output"] = data["output"].apply(clean_text)

# Step 4: Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

def tokenize_function(examples):
    return tokenizer(examples["input"], examples["output"], padding="max_length", truncation=True, max_length=512)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(data)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 5: Fine-tune the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Resize the token embeddings to account for the new padding token
model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./aki_gpt2_finetuned",  # Directory to save the model
    per_device_train_batch_size=4,     # Batch size
    num_train_epochs=3,                # Number of epochs
    logging_dir="./logs",              # Directory for logs
    save_steps=500,                    # Save model every 500 steps
    save_total_limit=2,                # Keep only the last 2 models
    evaluation_strategy="no",          # No evaluation during training
    logging_steps=100,                 # Log every 100 steps
    learning_rate=5e-5,                # Learning rate
    weight_decay=0.01,                 # Weight decay
    warmup_steps=100,                  # Warmup steps
    fp16=True,                         # Use mixed precision for faster training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Step 6: Save the fine-tuned model and tokenizer
model.save_pretrained("./aki_gpt2_finetuned")
tokenizer.save_pretrained("./aki_gpt2_finetuned")

print("Fine-tuning complete! Model saved to './aki_gpt2_finetuned'.")
