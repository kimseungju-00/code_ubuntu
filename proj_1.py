import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
import re

def load_chunked_dataset(file_path, num_chunks):
    chunked_datasets = []
    for i in range(num_chunks):
        chunk_file_path = f"{file_path[:-5]}_chunk_{i}.json"
        chunk_dataset = load_dataset('json', data_files=chunk_file_path)['train']
        chunked_datasets.append(chunk_dataset)
    return chunked_datasets

num_chunks = 236
chunked_datasets = load_chunked_dataset("/media/ksj/hd/code/merged_file.json", num_chunks)

new_dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

BASE_MODEL = "google/gemma-7b-it"

def generate_prompt(example):
    return [f"<bos><start_of_turn>user 다음 글을 요약해주세요: {example['document']}"]

lora_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.padding_side = 'right'

for i, train_data in enumerate(chunked_datasets):
    print(f"Training on chunk {i+1}/{num_chunks}")

    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # 토크나이저 전달
        train_dataset=train_data,
        max_seq_length=512,
        args=TrainingArguments(
            output_dir=f"outputs_{i}",
            max_steps=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            optim="paged_adamw_8bit",
            warmup_steps=0.03,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=2,
            push_to_hub=False,
            report_to='none',
        ),
        formatting_func=generate_prompt,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to('cuda')

    trainer.train()
    torch.cuda.empty_cache()

    model_save_path = f"gemma-7b-it-sum-ko_chunk_{i}"
    model = model.module if isinstance(model, torch.nn.DataParallel) else model
    model = model.merge_and_unload()
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)  # 토크나이저 저장

    if i < num_chunks - 1:
        model = AutoModelForCausalLM.from_pretrained(model_save_path, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)  # 토크나이저 로드

FINETUNE_MODEL = "gemma-7b-it-sum-ko"
model.save_pretrained(FINETUNE_MODEL)
tokenizer.save_pretrained(FINETUNE_MODEL)  # 최종 토크나이저 저장

finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL)
pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer)

test_doc = new_dataset['test']['document'][10]

def summarize_text(text):
    missing_entities = []
    dense_summary = f"이 텍스트는 {text}에 대해 설명합니다."
    summaries = [{"Missing_Entities": "", "Denser_Summary": dense_summary}]

    for _ in range(4):
        missing_entities = identify_missing_entities(text, summaries[-1]["Denser_Summary"])
        prev_summary = summaries[-1]["Denser_Summary"]
        dense_summary = generate_denser_summary(prev_summary, missing_entities)
        summaries.append({"Missing_Entities": "; ".join(missing_entities), "Denser_Summary": dense_summary})

    return summaries[-1]["Denser_Summary"]

def identify_missing_entities(text, summary):
    entities = re.findall(r'\w+', text)
    missing_entities = [entity for entity in entities if len(entity) <= 5 and entity.lower() not in summary.lower()]
    return missing_entities[:3]

def generate_denser_summary(prev_summary, missing_entities):
    for entity in missing_entities:
        prev_summary = re.sub(r'([\.\?\!])\s*', r'\1 ' + entity + ' ', prev_summary)
    return prev_summary

messages = [{"role": "user", "content": f"다음 글을 요약해주세요:\n\n{test_doc}"}]
prompt = pipe_finetuned.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe_finetuned(prompt, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, add_special_tokens=True)

final_summary = summarize_text(test_doc)
print(f"최종 요약: {final_summary}")
print(outputs[0]["generated_text"][len(prompt):])