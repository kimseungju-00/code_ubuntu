import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, PeftModel

# 두 개의 GPU를 선택합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 데이터셋 로드
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

# Hugging Face 모델 및 토크나이저 설정
BASE_MODEL = "google/gemma-7b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.padding_side = 'right'

# 데이터 전처리 및 모델 파이프라인 설정
doc = dataset['train']['document'][0]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

# 학습 데이터 포맷 설정
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

# LoRA 모델 설정
lora_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 모델을 병렬 GPU로 이동
model = torch.nn.DataParallel(model, device_ids=[0, 1])

# Trainer 설정 및 학습
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="outputs",
        max_steps=3000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
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
