import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Load annotated dataset from CSV file
df = pd.read_csv("task 1/data/annotated_dataset.csv")

# Convert pandas DataFrame to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define label-to-ID mapping
label_to_id = {"O": 0, "B-MNTN": 1, "I-MNTN": 2}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["Sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    for i, label in enumerate(examples["Annotation"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word indices
        label_ids = []
        label_tokens = label.split()  # Split annotation into label tokens

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore padding tokens
            else:
                if word_idx < len(label_tokens):  # Ensure valid index range
                    label_ids.append(label_to_id.get(label_tokens[word_idx], -100))
                else:
                    label_ids.append(-100)  # Default to ignored label

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Apply tokenization and label alignment
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Split dataset into training and testing sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Загрузка модели
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_to_id),  # Set number of unique labels
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
)

# Create Trainer instance for model training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
results = trainer.evaluate()
print(results)

# Save trained model and tokenizer
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
