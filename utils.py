import re
from typing import Optional
from datetime import datetime
import sqlglot
import matplotlib.pyplot as plt
from trl.trainer.sft_trainer import SFTTrainer


def setup_hf(hf_token: str):
    from huggingface_hub import login
    login(hf_token)


def create_conversation(sample):
    system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""
    user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.
    
    <SCHEMA>
    {context}
    </SCHEMA>
    
    <USER_QUERY>
    {question}
    </USER_QUERY>
    """
    
    return {
      "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
        {"role": "assistant", "content": sample["sql"]}
      ]
    }
    

def load_train_validate_dataset(n: Optional[int] = 10_000, validation_size: Optional[float] = 0.2):
    from datasets import load_dataset

    dataset_id = 'philschmid/gretel-synthetic-text-to-sql'
    dataset = load_dataset(dataset_id, split='train')
    dataset = dataset.select(range(n)) #type:ignore
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False) #type:ignore

    return dataset.train_test_split(test_size=validation_size, shuffle=False)


def load_test_dataset(n: Optional[int] = 1000):
    from datasets import load_dataset

    dataset_id = 'philschmid/gretel-synthetic-text-to-sql'
    dataset = load_dataset(dataset_id, split='test')

    return dataset.select(range(n)).map(create_conversation, remove_columns=dataset.features, batched=False) #type:ignore


def get_now():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


SQL_LANG_RE = re.compile(r'^\s*```(?:\s*sql\b|\s*SQL\b)?', re.IGNORECASE)

def _unwrap_markdown(md: str) -> str:
    fenced = re.findall(r'```([^`]*)```', md, flags=re.DOTALL)
    if fenced:
        fenced_with_lang = []
        blocks = []
        lines = md.splitlines(keepends=True)
        i = 0
        n = len(lines)
        while i < n:
            if lines[i].startswith("```"):
                header = lines[i].strip()
                i += 1
                content_lines = []
                while i < n and not lines[i].startswith("```"):
                    content_lines.append(lines[i])
                    i += 1
                if i < n and lines[i].startswith("```"):
                    i += 1
                blocks.append((header, ''.join(content_lines)))
            else:
                i += 1
        for head, content in blocks:
            if re.match(r'``` *sql\b', head, flags=re.IGNORECASE):
                fenced_with_lang.append(content)
        if fenced_with_lang:
            return "\n\n".join(fenced_with_lang).strip()
        # else return all fenced blocks joined
        if blocks:
            return "\n\n".join(c for _, c in blocks).strip()

    inline = re.findall(r'`([^`]+)`', md)
    if inline:
        return " ".join(inline).strip()

    return md.strip()


def normalize_sql(text: str) -> str:
    try:
        sql = _unwrap_markdown(text)
    except Exception:
        return text

    try:
        return sqlglot.parse_one(sql).sql(pretty=False)
    except Exception:
        return sql


def visualize_train_log(trainer: SFTTrainer, filename: str):
    log_history = trainer.state.log_history

    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log["epoch"] for log in log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

    plt.plot(epoch_train, train_losses, label="Training Loss")
    plt.plot(epoch_eval, eval_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

