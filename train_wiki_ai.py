import wikipedia
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import os

# ==========================================
# STEP 1: GATHER DATA FROM WIKIPEDIA
# ==========================================
print("Fetching Wikipedia data...")
wikipedia.set_lang("en")

# List of topics you want your AI to learn about
topics = ["Artificial intelligence", "Machine learning", "Deep learning", "Neural network"]
raw_text = ""

for topic in topics:
    try:
        print(f"Downloading: {topic}")
        page = wikipedia.page(topic)
        # Add the content of the page to our raw text
        raw_text += f"\n\n=== {topic} ===\n\n" + page.content
    except Exception as e:
        print(f"Could not fetch {topic}: {e}")

# Save the gathered data to a text file
train_file = "wiki_dataset.txt"
with open(train_file, "w", encoding="utf-8") as f:
    f.write(raw_text)
print(f"Data saved to {train_file}. Total length: {len(raw_text)} characters.\n")

# ==========================================
# STEP 2: PREPARE THE AI MODEL (GPT-2)
# ==========================================
print("Loading the tokenizer and model...")
# We use 'distilgpt2' because it is small and fast to train on a normal computer.
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# GPT2 tokenizer has no pad token by default; use eos_token as padding
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize the dataset using the datasets library
block_size = 128

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=block_size,
        padding="max_length",
    )

with open(train_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# Split text into chunks of block_size characters for training
chunks = [full_text[i:i + block_size * 4] for i in range(0, len(full_text), block_size * 4)]
raw_dataset = Dataset.from_dict({"text": chunks})
train_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_dataset = train_dataset.with_format("torch")

# A data collator formats the data for the training loop
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ==========================================
# STEP 3: TRAIN THE AI
# ==========================================
print("Setting up training...")
output_dir = "./my_wiki_ai"

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,  # How many times it reads the data (increase for better results)
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

print("Starting training! This might take a few minutes depending on your computer...")
trainer.train()

# Save the final trained model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training complete! Model saved to {output_dir}\n")

# ==========================================
# STEP 4: TEST YOUR NEW AI
# ==========================================
print("Testing the trained AI...")
# Load the model we just trained
trained_model = GPT2LMHeadModel.from_pretrained(output_dir)
trained_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Give the AI a starting prompt
prompt = "Artificial intelligence is "
input_ids = trained_tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = trained_model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
    do_sample=True
)

generated_text = trained_tokenizer.decode(output[0], skip_special_tokens=True)
print("\n--- AI Response ---")
print(generated_text)
print("-------------------")
