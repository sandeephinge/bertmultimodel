import json
import torch
from datasets import Dataset
from transformers import DistilBertTokenizer

# Load fixed dataset
file_path = "fixed_multi_model_hf.json"  # Ensure this path is correct
with open(file_path, "r") as f:
    data = json.load(f)

# Extract unique intents
intents = list(set(item["intent"] for item in data))
intent_label_map = {intent: i for i, intent in enumerate(intents)}

# Extract unique entity labels
unique_entity_labels = set()
for item in data:
    for entity in item["entities"]:
        unique_entity_labels.add(entity["entity"])
unique_entity_labels = sorted(unique_entity_labels)  # Ensure consistent order
# Ensure "B-" and "I-" prefixes are added for each entity type
entity_label_map = {"O": 0}  # "O" for non-entity words
for i, label in enumerate(unique_entity_labels, start=1):
    entity_label_map[f"B-{label}"] = i * 2 - 1  # Assign odd numbers
    entity_label_map[f"I-{label}"] = i * 2      # Assign even numbers

# Save intent label map
with open("intent_label_map.json", "w") as f:
    json.dump(intent_label_map, f)

# Save entity label map
with open("entity_label_map.json", "w") as f:
    json.dump(entity_label_map, f)

print(entity_label_map)  # Debug: Ensure all entity types exist 0  # "O" for non-entity words

# Load tokenizer
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

print("Loaded and Preprocessed the JSON Data....")

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True,
    )
    
    labels = ["O"] * len(tokenized_inputs["input_ids"])  # Default to "O"
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    for entity in example["entities"]:
        start, end, entity_type = entity["start"], entity["end"], entity["entity"]
        for idx, (start_offset, end_offset) in enumerate(offset_mapping):
            if start_offset == start:
                labels[idx] = f"B-{entity_type}"
            elif start_offset > start and end_offset <= end:
                labels[idx] = f"I-{entity_type}"
    
    tokenized_inputs["labels"] = [entity_label_map[label] for label in labels]
    tokenized_inputs["intent_label"] = intent_label_map[example["intent"]]
    
    return tokenized_inputs

# Apply tokenization
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(tokenize_and_align_labels, remove_columns=["text", "intent", "entities"])

print("Tokenized Text and Aligned Entity Labels...")

train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print("Prepared Data for Training...")

from transformers import DistilBertPreTrainedModel, DistilBertModel
from torch import nn

class DistilBERTMultiTask(DistilBertPreTrainedModel):
    def __init__(self, config, num_intents, num_entities):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        # Intent classification head
        self.intent_classifier = nn.Linear(config.hidden_size, num_intents)

        # Token classification head
        self.token_classifier = nn.Linear(config.hidden_size, num_entities)

        # Loss functions
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None, intent_label=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state

        # Compute intent logits
        pooled_output = hidden_state[:, 0, :]  # CLS token
        intent_logits = self.intent_classifier(pooled_output)

        # Compute token classification logits
        token_logits = self.token_classifier(hidden_state)

        loss = None
        if labels is not None and intent_label is not None:
            intent_loss = self.intent_loss_fn(intent_logits, intent_label)
            token_loss = self.token_loss_fn(token_logits.view(-1, token_logits.shape[-1]), labels.view(-1))
            loss = intent_loss + token_loss  # Joint loss

        return {"loss": loss, "intent_logits": intent_logits, "token_logits": token_logits}

print("Defined a Multi-Task DistilBERT Model...")

from transformers import TrainingArguments, Trainer

num_intents = len(intent_label_map)
num_entities = len(entity_label_map)

# Load model
model = DistilBERTMultiTask.from_pretrained("distilbert-base-uncased", num_intents=num_intents, num_entities=num_entities)

# Training arguments
training_args = TrainingArguments(
    output_dir="./multi_task_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
trainer.train()

print("Trained the Model Using Hugging Face’s Trainer...")

model.save_pretrained("multi_task_model")
tokenizer.save_pretrained("multi_task_model")

print("Saved the trained model...")

