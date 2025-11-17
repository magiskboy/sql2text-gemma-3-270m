import os
import json
import click
from tqdm import tqdm
import utils


base_model = 'google/gemma-3-270m-it'


@click.group
def main():
    ...


@main.command()
@click.option(
    '--lr', 
    type=click.FLOAT, 
    default=5e-5, 
    help='Learning rate dùng cho quá trình huấn luyện.'
)
@click.option(
    '--epoch', 
    type=click.INT, 
    default=5, 
    help='Số lượng epoch để train.'
)
@click.option(
    '--lora-rank', 
    type=click.INT, 
    default=8, 
    help='Giá trị rank của LoRA (r).'
)
@click.option(
    '--lora-alpha', 
    type=click.INT, 
    default=16, 
    help='LoRA alpha – hệ số scale cho update của LoRA.'
)
@click.option(
    '--lora-dropout', 
    type=click.FLOAT, 
    default=0.05,
    help='Tỷ lệ dropout áp dụng cho các module LoRA.'
)
@click.option(
    '--lora-modules', 
    type=click.STRING, 
    default='q_proj,k_proj,v_proj,o_proj',
    help='Danh sách module áp dụng LoRA, phân tách bằng dấu phẩy.'
)
@click.option(
    '--validation-size', 
    type=click.FLOAT, 
    default=0.2,
    help='Tỷ lệ dữ liệu validation so với toàn bộ dataset.'
)
@click.option(
    '--hf-token', 
    type=click.STRING,
    help='HuggingFace token để truy cập và tải model/dataset.'
)
@click.option(
    '--checkpoint-dir', 
    type=click.STRING, 
    default='checkpoints',
    help='Thư mục lưu checkpoint trong quá trình training.'
)
@click.option(
    '--optimizer', 
    type=click.STRING, 
    default='adamw_torch_fused',
    help='Tên optimizer sử dụng (vd: adamw_torch, adamw_torch_fused...).'
)
def train(
    lr: float,
    epoch: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_modules: str,
    validation_size: float,
    hf_token: str,
    checkpoint_dir: str,
    optimizer: str,
):
    import torch
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if hf_token:
        utils.setup_hf(hf_token)

    dataset = utils.load_train_validate_dataset(validation_size=validation_size)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    torch_dtype = model.dtype

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] if lora_modules is None else lora_modules.split(','),
        lora_dropout=lora_dropout,
        bias="none",
    )

    args = SFTConfig(
        output_dir=checkpoint_dir,
        max_length=512,                         # max sequence length for model and packing of the dataset
        packing=False,                          # Groups multiple samples in the dataset into a single sequence
        num_train_epochs=epoch,
        per_device_train_batch_size=4,
        gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
        optim=optimizer,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=lr,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="tensorboard",
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        }
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model()


@main.command()
@click.option(
    '--hf-token',
    type=click.STRING,
    help='HuggingFace token dùng để tải model/dataset nếu cần.'
)
@click.option(
    '--checkpoint-dir',
    type=click.STRING,
    default='checkpoints',
    help='Thư mục chứa các checkpoint để load model khi đánh giá.'
)
@click.option(
    '--max-tokens',
    type=click.INT,
    default=4096,
    help='Giới hạn số lượng token tối đa cho mỗi sample khi eval.'
)
@click.option(
    '--report-dir',
    type=click.STRING,
    default='report',
    help='Thư mục đầu ra để lưu kết quả đánh giá và báo cáo.'
)
def eval(
    hf_token: str,
    checkpoint_dir: str,
    max_tokens: int,
    report_dir: str,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    report_dir = report_dir or 'report'
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    if hf_token:
        utils.setup_hf(hf_token)

    dataset = utils.load_test_dataset()

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Warm up time
    item = dataset[0]
    prompt = pipe.tokenizer.apply_chat_template(item["messages"][:2], tokenize=False, add_generation_prompt=True) #type:ignore
    outputs = pipe(prompt, max_new_tokens=max_tokens, disable_compile=True) #type:ignore
    print(f'Prompt\n{prompt}')
    print(f'Expected value\n{item["messages"][2]["content"]}')
    print(f'Generated\n{outputs[0]["generated_text"][len(prompt):].strip()}')


    reports = {
        "n_samples": 0,
        "pass@1": 0,
        "details": [],
    }

    for item in tqdm(dataset, desc='Generate'):
        expected = item["messages"][2]["content"] #type:ignore
        prompt = pipe.tokenizer.apply_chat_template(item["messages"][:2], tokenize=False, add_generation_prompt=True) #type:ignore

        try:
            outputs = pipe(prompt, max_new_tokens=4096, disable_compile=True) #type:ignore
            predicted = outputs[0]["generated_text"][len(prompt):].strip()

            reports['details'].append({
              "predicted": predicted,
              "expected": expected,
              "prompt": prompt,
              "pass": False,
            })

            reports['n_samples'] += 1

        except Exception as e:
            reports['details'].append({
                "error": str(e),
                "prompt": prompt,
                "pass": False,
                "expected": expected,
            })

    for item in tqdm(reports['details'], desc='Evaluate'):
        if "error" in item:
            continue
        is_passed = utils.normalize_sql(item['expected']) == utils.normalize_sql(item['predicted'])
        item['pass'] = is_passed
        if is_passed:
            reports['pass@1'] += 1

    p = reports['pass@1'] / reports['n_samples'] * 100
    print('Report')
    print('Number of samples', reports['n_samples'])
    print('Pass@1', reports['pass@1'])
    print(f"Percentage: {p:.2}")

    report_filename = os.path.join(report_dir, f'report_{utils.get_now()}.json')
    with open(report_filename, 'w') as report_file:
        json.dump(reports, report_file, indent=4)
    
    click.echo(f"Report was exported to {report_filename}")


if __name__ == '__main__':
    main()

