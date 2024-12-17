import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset, load_dataset, DatasetDict
from trl import DataCollatorForCompletionOnlyLM
import json  # Import the json module
from peft import get_peft_model, LoraConfig, TaskType
import os
from huggingface_hub import login

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

   # huggingface-cli login, run in terminal to get access to model
   # or uncomment this code with your own key
#login(token="")


# Step 2: Define configurations and load model/tokenizer
checkpoint = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding="right")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    max_memory={0: "8GB"}
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    model = model.to('cuda') #enbale CUDA

# Add padding token to tokenizer (necessary for fine-tuning)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

# Step 3: Prepare dataset
# Example dataset: Multi-turn conversations
dataset = load_dataset('json', data_files='data/MLDial.json')

dataset = dataset["train"].train_test_split(test_size=0.1)

train_test_dict = {
    'train': dataset['train'],
    'validation': dataset['test']  # Rename the split by creating a new dictionary
}

dataset = DatasetDict(train_test_dict)


def tokenize_function(samples):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for conversation in samples["conversation_turns"]:
        formatted = ""
        for turn in conversation:
            if turn["role"] == "user":
                formatted += f"<s><INST> {turn['content']} </INST>"
            elif turn["role"] == "assistant":
                formatted += f" {turn['content']} </s>"
        
        # Tokenize the formatted conversation
        tokenized = tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=512,  
            return_tensors="pt"
        )
        
        # Append input IDs, attention mask, and labels
        all_input_ids.append(tokenized["input_ids"][0])
        all_attention_mask.append(tokenized["attention_mask"][0])
        labels = tokenized["input_ids"][0].clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
        all_labels.append(labels)
    
    return {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "labels": torch.stack(all_labels)
    }

# Apply tokenization to dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Step 4: Create data collator
instruction_template = "[INST]"
response_template = "[/INST]"


# Step 5: Create DataLoader for training
dataloader = torch.utils.data.DataLoader(
    tokenized_dataset["train"],
    batch_size=1,  # Adjust batch size as needed
)

# Step 6: Check a batch (illustration)
for batch in dataloader:
    print(batch)
    break

# Step 7: Define training procedure 
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)
trainer.train()

model.save_pretrained("./fine-tuned-llama2-MLDial")