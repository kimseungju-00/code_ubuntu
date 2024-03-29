import os

# 사용할 GPU를 선택합니다. 예를 들어, 첫 번째 GPU와 두 번째 GPU를 선택하려면 아래와 같이 설정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from datasets import load_dataset
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

BASE_MODEL = "google/gemma-7b-it"

#model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
#tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)

doc = dataset['train']['document'][0]

def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['document'])):
        prompt_list.append(r"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['document'][i], example['summary'][i]))
    return prompt_list

train_data = dataset['train']
#print(generate_prompt(train_data[:1])[0])

lora_config = LoraConfig(
    r=6,
    lora_alpha = 8,
    lora_dropout = 0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.padding_side = 'right'

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="outputs",
#        num_train_epochs = 1,
        max_steps=3000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        warmup_steps=0.03,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
    ),
    peft_config=lora_config,
    formatting_func=generate_prompt,
)

trainer.train()

ADAPTER_MODEL = "lora_adapter"

trainer.model.save_pretrained(ADAPTER_MODEL)

model3 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
model3 = PeftModel.from_pretrained(model3, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

model3 = model3.merge_and_unload()
model3.save_pretrained('gemma-7b-it-sum-ko')

BASE_MODEL = "google/gemma-7b-it"
FINETUNE_MODEL = "./gemma-7b-it-sum-ko"

finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.padding_side = 'right'

pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=512)

doc = dataset['test']['document'][10]

messages = [
    {
        "role": "user",
        "content": "다음 글을 요약해주세요:\n\n{}".format(doc)
    }
]
prompt = pipe_finetuned.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe_finetuned(
    prompt,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.95,
    add_special_tokens=True
)
print(outputs[0]["generated_text"][len(prompt):])
