from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch
import datasets
from transformers import pipeline

BASE_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
SYSTEM_PROMPT = "You are student and you task is to answer the question based on provided context.\
You will return the answer in the following format </json>ANSWER</json>. Do not add any futher explanation."


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto",)



def format_dataset(sample,is_answer:bool=True):
    prompt = f"{SYSTEM_PROMPT} ### Question: {sample['question']} ### Context: {sample['context']} ### Answer:"

    if is_answer:
        return  {"prompt": prompt,
            "completion": f"</json>{sample['answer']}</json>" ,
            }
    return {"prompt": prompt}

def get_train_data(size):

    # Import the data
    dataset = datasets.load_dataset("rajpurkar/squad",split="train")
    dataset = dataset.select(range(size))
    dataset = dataset.map(lambda x: {'answer': x['answers']['text'][0]})
    dataset = dataset.remove_columns(['title','answers'])
    dataset = dataset.map(format_dataset,fn_kwargs={"is_answer":True})
    dataset = dataset.remove_columns(['answer','question','context'])
    return dataset

train_dataset = get_train_data(size=3000)



###############################
#  Configure Model
##############################
peft_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    # Model Config
    max_length=1024,
    max_steps=1000,
    # eval_strategy="steps",
    # eval_steps=50,
    bf16=True,  # Use bfloat16 precision
    # Batch Size settings
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch
    # Saving settings
    output_dir='./temp',
    logging_steps=10,
    save_steps=100,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

###############################
#  Train Model
##############################
trainer.train()


###############################
#  Evaluate Model
##############################
def get_test_data():
    test_dataset = datasets.load_dataset("rajpurkar/squad",split="train")
    test_dataset = test_dataset.select(range(6000,6004))
    test_dataset = test_dataset.map(lambda x: {'answer': x['answers']['text'][0]})
    test_dataset = test_dataset.remove_columns(['title','answers'])
    test_dataset = test_dataset.map(format_dataset,fn_kwargs={"is_answer":False})
    test_dataset = test_dataset.remove_columns(['question','context'])

test_dataset = get_test_data()

# Define generator
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
generated_text = generator(test_dataset[0]['prompt'])

print(generated_text)
