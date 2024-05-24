import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
from datasets import load_dataset

# GPU 설정: 사용할 GPU를 지정 (여기서는 GPU 0, 1 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 사용할 기본 모델 지정
BASE_MODEL = "google/gemma-7b-it"

# 사전 학습된 모델을 로드하고 4bit 양자화 설정 적용
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,  # 4bit 양자화를 사용하여 모델 로드
        bnb_4bit_quant_type="nf4",  # 양자화 유형 설정
        bnb_4bit_compute_dtype=torch.float16  # 계산 시 사용할 데이터 타입 설정
    )
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'right'  # 패딩을 오른쪽에 추가하도록 설정

# JSON 파일로부터 데이터셋 로드
dataset = load_dataset('json', data_files='/media/ksj/hd/merged_file.json')

# 데이터셋을 훈련 세트와 테스트 세트로 분할 (테스트 세트 비율: 20%)
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_data = split_dataset['train']
val_data = split_dataset['test']

# 데이터셋 크기 확인
print(f"Train dataset size: {len(train_data)}")
print(f"Validation dataset size: {len(val_data)}")

# 모델이 DataParallel로 감싸져 있는 경우 원본 모델을 추출
if isinstance(model, torch.nn.DataParallel):
    model = model.module

# LoRA 구성 변경
model = get_peft_model(
    model,
    LoraConfig(
        r=8,  # 랭크 값 (저장할 저차원 행렬의 랭크)
        lora_alpha=16,  # LoRA 계수 (모델 학습 속도에 영향)
        lora_dropout=0.1,  # 드롭아웃 비율 (과적합 방지)
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # 적용할 모듈 목록
        task_type="CAUSAL_LM",  # 작업 유형 (언어 모델링)
    )
)

# 커스텀 콜백 클래스를 정의하여 로그를 파일로 저장
class LoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(logs) + "\n")

# 로그 파일 경로 설정
log_file = "training_logs.txt"

# SFTTrainer 설정: 모델 훈련을 위한 다양한 매개변수와 데이터셋 지정
training_args = TrainingArguments(
    output_dir="outputs",  # 출력 디렉토리
    max_steps=200,  # 최대 훈련 스텝 수
    per_device_train_batch_size=1,  # 배치 크기 줄이기
    gradient_accumulation_steps=32,  # 기울기 누적 스텝 수 증가 (메모리 절약)
    optim="paged_adamw_8bit",  # 옵티마이저 설정 (8bit AdamW)
    warmup_steps=0.03,  # 워밍업 스텝 비율 (학습 초기에 학습률을 점차 증가)
    learning_rate=1e-4,  # 학습률
    fp16=True,  # 혼합 정밀도 학습 (FP16)
    logging_steps=2,  # 로그를 기록할 스텝 간격
    push_to_hub=False,  # 모델을 허브에 푸시할지 여부
    report_to='none',  # 로깅 리포트 대상 (여기서는 사용 안 함)
    weight_decay=0.01,  # 가중치 감쇠 (정규화 기법)
    evaluation_strategy="steps",  # 평가 전략 (스텝 간격으로 평가)
    eval_steps=2,  # 평가 스텝 간격
    load_best_model_at_end=True,  # 훈련 후 가장 좋은 모델 로드
    metric_for_best_model="eval_loss",  # 최적 모델을 선택할 메트릭 (검증 손실)
    greater_is_better=False,  # 메트릭이 작을수록 좋음
    logging_dir="logs",  # 로그 디렉토리 추가
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,  # 훈련 데이터셋
    eval_dataset=val_data,  # 검증 데이터셋
    max_seq_length=128,  # 최대 시퀀스 길이 줄이기
    args=training_args,
    formatting_func=lambda example: [f"<bos><start_of_turn>user 다음 글을 요약해주세요: {example['document']}"],  # 입력 형식 정의
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LoggingCallback(log_file)],  # 조기 종료 및 로그 콜백 추가
)

# 여러 GPU를 사용하는 경우 모델을 DataParallel로 래핑
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# 모델 훈련
trainer.train()

# 모델 저장
trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")
