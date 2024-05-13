import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 환경 변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 모델 및 토크나이저 로드
BASE_MODEL = "google/gemma-7b-it"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 토크나이저의 모든 속성을 검사하여 직렬화할 수 없는 속성을 찾기
def check_tokenizer_attributes(tokenizer):
    for attr in dir(tokenizer):
        attr_value = getattr(tokenizer, attr)
        # callable은 함수, 메서드 등 호출 가능한 객체를 확인하는 파이썬 내장 함수입니다.
        if callable(attr_value) or isinstance(attr_value, torch.nn.Module):
            print(f"{attr} is a callable or a torch module and might not be serializable.")

# 토크나이저 속성 검사 실행
check_tokenizer_attributes(tokenizer)
