"""Утилиты для лёгкого дообучения каузальных LLM по пользовательским данным"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from models import FineTuneExample

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "{task_section}"
    "### Instruction:\n{instruction}\n"
    "{labels_section}"
    "{context_section}"
    "### Answer:\n{response}\n"
)


@dataclass
class _TrainingConfig:
    """Все параметры, необходимые для запуска обучения"""

    base_model_name: str
    output_dir: str
    run_name: str
    learning_rate: float
    num_epochs: int
    max_seq_length: int
    gradient_accumulation_steps: int
    trust_remote_code: bool


class _InstructionDataset(Dataset):
    """Простой датасет в памяти для задач instruction tuning"""

    def __init__(self, tokenizer, examples: Iterable[FineTuneExample], max_length: int) -> None:
        formatted_texts: List[str] = []
        for sample in examples:
            task_section = ""
            if sample.task:
                task_section = f"### Task:\n{sample.task.strip()}\n"
            labels_section = ""
            if sample.labels:
                cleaned_labels = [label.strip() for label in sample.labels if label and label.strip()]
                if cleaned_labels:
                    labels_section = f"### Labels:\n{', '.join(cleaned_labels)}\n"
            context_parts = []
            if sample.context:
                context_parts.append(sample.context.strip())
            if sample.metadata:
                try:
                    context_parts.append(json.dumps(sample.metadata, ensure_ascii=False))
                except (TypeError, ValueError):
                    context_parts.append(str(sample.metadata))
            context_section = ""
            if context_parts:
                joined_context = "\n\n".join(context_parts)
                context_section = f"### Context:\n{joined_context}\n"
            prompt = PROMPT_TEMPLATE.format(
                task_section=task_section,
                instruction=sample.instruction.strip(),
                labels_section=labels_section,
                context_section=context_section,
                response=sample.response.strip(),
            )
            formatted_texts.append(prompt)
        encodings = tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class _TrainerLoggingCallback(TrainerCallback):
    """Простой callback, пробрасывающий логи Trainer в логгер приложения"""

    def on_train_begin(self, args, state, control, **kwargs):  # pylint: disable=unused-argument
        max_steps = getattr(state, "max_steps", "unknown")
        logger.info(
            "Trainer run started: total_steps≈%s, gradient_accumulation=%s",
            max_steps,
            args.gradient_accumulation_steps,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):  # pylint: disable=unused-argument
        if not logs:
            return
        filtered = {k: v for k, v in logs.items() if not k.startswith("total_flos")}
        logger.info(
            "Trainer log | step=%s epoch=%s metrics=%s",
            getattr(state, "global_step", None),
            getattr(state, "epoch", None),
            filtered,
        )


def _summarize_examples(examples: List[FineTuneExample]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for sample in examples:
        key = (sample.task or "unknown").strip() or "unknown"
        summary[key] = summary.get(key, 0) + 1
    return summary


def _prepare_tokenizer(base_model_name: str, trust_remote_code: bool, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def _prepare_model(base_model_name: str, tokenizer, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model


def fine_tune_local_model(
    *,
    examples: List[FineTuneExample],
    base_model_name: str,
    output_dir: str,
    run_name: Optional[str] = None,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 1,
    trust_remote_code: bool = True,
) -> str:
    """
    Запускает дообучение каузальной LLM на наборах instruction/response, переданных пользователем

    Returns:
        Строку с путём к директории, где сохранены артефакты дообученной модели
    """
    if not examples:
        raise ValueError("Не переданы примеры для дообучения.")

    summary = _summarize_examples(examples)
    logger.info(
        "Подготовка дообучения: %s примеров. Распределение по задачам: %s",
        len(examples),
        summary,
    )
    first_example = examples[0]
    logger.info(
        "Пример #1 | task=%s | instruction='%s' | response='%s'",
        first_example.task or "unknown",
        first_example.instruction[:200].replace("\n", " "),
        first_example.response[:200].replace("\n", " "),
    )

    os.makedirs(output_dir, exist_ok=True)
    resolved_run_name = run_name or datetime.utcnow().strftime("user-model-%Y%m%d-%H%M%S")
    run_path = os.path.join(output_dir, resolved_run_name)
    os.makedirs(run_path, exist_ok=True)

    config = _TrainingConfig(
        base_model_name=base_model_name,
        output_dir=run_path,
        run_name=resolved_run_name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        trust_remote_code=trust_remote_code,
    )

    logger.info(
        "Старт дообучения модели %s на %s примерах. lr=%s epochs=%s batch_size=1 grad_accum=%s. Директория: %s",
        config.base_model_name,
        len(examples),
        config.learning_rate,
        config.num_epochs,
        config.gradient_accumulation_steps,
        config.output_dir,
    )

    tokenizer = _prepare_tokenizer(config.base_model_name, config.trust_remote_code, config.max_seq_length)
    model = _prepare_model(config.base_model_name, tokenizer, config.trust_remote_code)

    if torch.cuda.is_available():
        device = "cuda"
        device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        device = "cpu"
        device_names = []
    logger.info(
        "Аппаратная конфигурация дообучения: device=%s, cuda_available=%s, cuda_devices=%s",
        device,
        torch.cuda.is_available(),
        device_names or "none",
    )

    dataset = _InstructionDataset(tokenizer=tokenizer, examples=examples, max_length=config.max_seq_length)
    logger.info(
        "Датасет токенизирован: %s примеров, max_seq_length=%s, vocab=%s.",
        len(dataset),
        config.max_seq_length,
        len(tokenizer),
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.add_callback(_TrainerLoggingCallback())

    trainer.train()
    state = trainer.state
    global_step = getattr(state, "global_step", None) if state else None
    epoch = getattr(state, "epoch", None) if state else None
    num_samples = getattr(state, "num_train_samples", None) if state else None
    train_runtime = getattr(state, "train_runtime", None) if state else None
    logger.info(
        "Обучение завершено: steps=%s epochs=%s samples=%s runtime=%s.",
        global_step,
        epoch,
        num_samples,
        f"{train_runtime:.2f}s" if isinstance(train_runtime, (int, float)) else train_runtime,
    )
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info("Дообучение завершено. Артефакты сохранены в %s", config.output_dir)
    return config.output_dir
