import os

# 사용할 GPU를 선택합니다. 예를 들어, 첫 번째 GPU와 두 번째 GPU를 선택하려면 아래와 같이 설정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 나머지 코드는 그대로 유지합니다.
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from datasets import load_dataset
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

BASE_MODEL = "google/gemma-7b-it"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)

doc = dataset['train']['document'][0]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

messages = [
    {
        "role": "user",
        "content": "다음 글을 요약해주세요:\n\n{}".format(doc)
    }
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe(
    prompt,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.95,
    add_special_tokens=True
)

print(outputs[0]["generated_text"][len(prompt):])
