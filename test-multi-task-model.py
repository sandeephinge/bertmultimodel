import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertPreTrainedModel, DistilBertModel
from torch import nn

# ✅ Define Multi-Task DistilBERT Model (Same as Training)
class DistilBERTMultiTask(DistilBertPreTrainedModel):
    def __init__(self, config, num_intents, num_entities):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        # Intent classification head
        self.intent_classifier = nn.Linear(config.hidden_size, num_intents)

        # Token classification head
        self.token_classifier = nn.Linear(config.hidden_size, num_entities)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state

        # Compute intent logits
        pooled_output = hidden_state[:, 0, :]
        intent_logits = self.intent_classifier(pooled_output)

        # Compute token classification logits
        token_logits = self.token_classifier(hidden_state)

        return {"intent_logits": intent_logits, "token_logits": token_logits}

# ✅ Load Model and Tokenizer
model_path = "multi_task_model"

print("Loading model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# ✅ Load intent label map and REVERSE it (so numbers map to intent names)
with open(f"intent_label_map.json", "r") as f:
    intent_label_map = json.load(f)
intent_label_map = {v: k for k, v in intent_label_map.items()}  # Reverse the mapping

# ✅ Load entity label map and ensure integer keys
with open(f"entity_label_map.json", "r") as f:
    entity_label_map = json.load(f)

# ✅ Convert only numeric keys, keep "O" as it is
entity_label_map = {v: k for k, v in entity_label_map.items()} # Convert keys to integers

# ✅ Load trained model with correct label sizes
num_intents = len(intent_label_map)
num_entities = len(entity_label_map)
model = DistilBERTMultiTask.from_pretrained(model_path, num_intents=num_intents, num_entities=num_entities)
model.eval()

print("Model loaded successfully!")

def predict_intent_and_entities(text):
    """Predicts the intent and extracts named entities from a given sentence."""
    
    # ✅ Tokenize text and return offsets for entity extraction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()  # Get character positions

    with torch.no_grad():
        outputs = model(**inputs)

    # ✅ Intent Prediction
    intent_pred = torch.argmax(outputs["intent_logits"], dim=-1).item()
    intent = intent_label_map.get(intent_pred, "Unknown Intent")

    # ✅ Token Predictions (Convert model output to entity labels)
    token_preds = torch.argmax(outputs["token_logits"], dim=-1).squeeze().tolist()
    if isinstance(token_preds, int):  # Ensure token_preds is a list
        token_preds = [token_preds]

    entity_preds = [entity_label_map.get(i, "O") for i in token_preds]  # Convert numbers to labels safely

    # ✅ Extract Entity Values
    entities = []
    current_entity = None
    current_value = ""
    start_pos = None

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    for i, label in enumerate(entity_preds):
        word_start, word_end = offset_mapping[i]  # Get original word position
        word = text[word_start:word_end]  # Extract the actual word

        if label.startswith("B-"):  # Beginning of a new entity
            if current_entity:  # Save the previous entity before starting a new one
                entities.append({"entity": current_entity, "value": current_value.strip(), "start": start_pos, "end": word_start})

            current_entity = label[2:]  # Remove "B-" prefix
            current_value = word
            start_pos = word_start

        elif label.startswith("I-") and current_entity == label[2:]:  # Inside an existing entity
            current_value += " " + word

        else:  # If "O" or a different entity starts, save the previous entity
            if current_entity:
                entities.append({"entity": current_entity, "value": current_value.strip(), "start": start_pos, "end": word_start})
                current_entity = None
                current_value = ""

    # Save the last entity if it exists
    if current_entity:
        entities.append({"entity": current_entity, "value": current_value.strip(), "start": start_pos, "end": word_end})

    return {"intent": intent, "entities": entities}

# ✅ Run a Test Query
if __name__ == "__main__":
    test_sentence = "give me salary slip for previous month"
    print(f"\n🔍 Testing Model on: '{test_sentence}'")
    result = predict_intent_and_entities(test_sentence)
    print("\n✅ Prediction Result:")
    print(result)

