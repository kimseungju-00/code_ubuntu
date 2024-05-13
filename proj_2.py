import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_chunked_dataset(file_path, num_chunks):
    chunked_datasets = []
    for i in range(num_chunks):
        chunk_file_path = f"{file_path[:-5]}_chunk_{i}.json"
        chunk_dataset = load_dataset('json', data_files=chunk_file_path)['train']
        chunked_datasets.append(chunk_dataset)
    return chunked_datasets

num_chunks = 236
chunked_datasets = load_chunked_dataset("/media/ksj/hd/code/merged_file.json", num_chunks)

BASE_MODEL = "google/gemma-7b-it"
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'right'  # Ensure tokenizer padding side is 'right'

def save_custom_pretrained(tokenizer, save_directory):
    tokenizer_config = {
        'vocab_size': tokenizer.vocab_size,
        'max_len': tokenizer.model_max_length,
        'padding_side': tokenizer.padding_side,
        'model_input_names': tokenizer.model_input_names
    }
    out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        f.write(out_str)

for i, train_data in enumerate(chunked_datasets):
    print(f"Training on chunk {i+1}/{num_chunks}")
    
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # DataParallel에서 원본 모델 추출
    
    model = get_peft_model(
        model,
        LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        max_seq_length=256,
        args=TrainingArguments(
            output_dir=f"outputs_{i}",
            max_steps=5,
            per_device_train_batch_size=1,  # 배치 크기를 더 줄여볼 수 있습니다. 예: 0.5 or less
            gradient_accumulation_steps=32,
            optim="paged_adamw_8bit",
            warmup_steps=0.03,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=2,
            push_to_hub=False,
            report_to='none',
        ),
        formatting_func=lambda example: [f"<bos><start_of_turn>user 다음 글을 요약해주세요: {example['document']}"],
    )
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to('cuda')
    
    trainer.train()
    torch.cuda.empty_cache()
    
    # 10개의 청크마다 모델 저장
    if (i + 1) % 10 == 0:
        model_save_path = f"gemma-7b-it-sum-ko_chunk_{i}"
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(model_save_path)
        else:
            model.save_pretrained(model_save_path)
        save_custom_pretrained(tokenizer, model_save_path)

# 학습이 완료된 후 최종 모델 저장
model_save_path = "gemma-7b-it-sum-ko"
if isinstance(model, torch.nn.DataParallel):
    model.module.save_pretrained(model_save_path)
else:
    model.save_pretrained(model_save_path)
save_custom_pretrained(tokenizer, model_save_path)