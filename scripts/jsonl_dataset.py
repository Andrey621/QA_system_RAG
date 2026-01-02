"""Утилиты для маркетинговых JSONL-датасетов и их конвертации в формат Яндекс-чата."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Default system prompt used in Yandex-style dataset generation.
DEFAULT_SYSTEM_PROMPT = (
    "Ты — эксперт по латеральному маркетингу, вертикальному маркетингу и "
    "разработке категорийных инноваций. Отвечай кратко, четко и по делу, без воды. "
    "По возможности опирайся на логику и терминологию книги «Латеральный маркетинг» "
    "Филиппа Котлера и Фернандо Триаса де Беса. Если вопрос про классификацию или "
    "тип инновации, давай прямой однозначный ответ без лишних пояснений."
)


def clean_value(value: Any) -> str:
    """Возвращает строку без крайних пробелов вне зависимости от исходного типа."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def load_examples_from_jsonl(
    file_path: Path,
    *,
    split: Optional[str],
    prompt_key: str,
    response_key: str,
    evidence_key: Optional[str],
    task_key: Optional[str],
    labels_key: Optional[str],
    meta_key: Optional[str],
    split_key: Optional[str],
    include_meta: bool,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """Преобразует строки JSONL в формат instruction/response, понятный FastAPI."""

    rows = _read_jsonl(file_path)
    examples: List[Dict[str, Any]] = []
    normalized_split = (split or "").lower()

    for row in rows:
        if normalized_split and normalized_split != "all":
            value = clean_value(row.get(split_key or "split")) if split_key else ""
            if value.lower() != normalized_split:
                continue

        instruction = clean_value(row.get(prompt_key, ""))
        response = clean_value(row.get(response_key, ""))
        if not instruction or not response:
            continue

        example: Dict[str, Any] = {"instruction": instruction, "response": response}
        context_parts: List[str] = []

        if evidence_key:
            evidence_value = row.get(evidence_key)
            if isinstance(evidence_value, list):
                context_parts.extend([clean_value(ev) for ev in evidence_value if clean_value(ev)])
            elif isinstance(evidence_value, str):
                cleaned = clean_value(evidence_value)
                if cleaned:
                    context_parts.append(cleaned)

        if include_meta and meta_key:
            meta_value = row.get(meta_key)
            if isinstance(meta_value, dict) and meta_value:
                context_parts.append(json.dumps(meta_value, ensure_ascii=False))

        if context_parts:
            example["context"] = "\n".join(context_parts)

        if task_key:
            task_value = clean_value(row.get(task_key, ""))
            if task_value:
                example["task"] = task_value

        if labels_key:
            labels_value = row.get(labels_key)
            if isinstance(labels_value, list):
                cleaned_labels = [clean_value(label) for label in labels_value if clean_value(label)]
                if cleaned_labels:
                    example["labels"] = cleaned_labels

        if meta_key:
            meta_value = row.get(meta_key)
            if isinstance(meta_value, dict) and meta_value:
                example["metadata"] = meta_value

        examples.append(example)

        if limit and len(examples) >= limit:
            break

    if not examples:
        raise ValueError("Не удалось извлечь данные из JSONL файла.")
    return examples


def _build_context_parts(
    row: Dict[str, Any],
    *,
    evidence_key: Optional[str],
    meta_key: Optional[str],
    include_evidence: bool,
    include_meta: bool,
) -> List[str]:
    """Собирает дополнительные блоки контекста: доказательства и метаданные."""
    context_parts: List[str] = []
    if include_evidence and evidence_key:
        evidence_value = row.get(evidence_key)
        if isinstance(evidence_value, list):
            context_parts.extend([clean_value(ev) for ev in evidence_value if clean_value(ev)])
        elif isinstance(evidence_value, str):
            cleaned = clean_value(evidence_value)
            if cleaned:
                context_parts.append(cleaned)
    if include_meta and meta_key:
        meta_value = row.get(meta_key)
        if isinstance(meta_value, dict) and meta_value:
            context_parts.append(json.dumps(meta_value, ensure_ascii=False))
    return [part for part in context_parts if part]


def convert_to_yandex_format(
    source_path: Path,
    output_path: Path,
    *,
    split: Optional[str] = None,
    prompt_key: str = "prompt",
    response_key: str = "response",
    task_key: str = "task",
    evidence_key: str = "evidence",
    labels_key: str = "labels",
    meta_key: str = "meta",
    split_key: str = "split",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_evidence: bool = False,
    include_meta: bool = False,
    include_labels: bool = False,
    limit: Optional[int] = None,
) -> int:
    """
    Конвертирует строки датасета во внешний формат JSONL Яндекса:
        {"request": [{"role": "system", ...}, {"role": "user", ...}], "response": "..."}.

    Возвращает количество записанных строк.
    """

    rows = _read_jsonl(source_path)
    normalized_split = (split or "").lower()
    total_written = 0

    system_message = system_prompt.strip()
    if not system_message:
        raise ValueError("System prompt must not be empty.")

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if normalized_split and normalized_split != "all":
                value = clean_value(row.get(split_key or "split")) if split_key else ""
                if value.lower() != normalized_split:
                    continue

            prompt = clean_value(row.get(prompt_key, ""))
            response = row.get(response_key)
            response_str = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)

            if not prompt or not response_str:
                continue

            task_value = clean_value(row.get(task_key, "")).lower() if task_key else ""
            task_text = task_value if task_value else "unknown"

            user_lines: List[str] = [f"Задача: {task_text}.", f"Вопрос: {prompt}"]

            if include_labels and labels_key:
                labels_value = row.get(labels_key)
                if isinstance(labels_value, list):
                    cleaned_labels = [clean_value(label) for label in labels_value if clean_value(label)]
                    if cleaned_labels:
                        user_lines.append(f"Варианты ответа: {', '.join(cleaned_labels)}")

            context_parts = _build_context_parts(
                row,
                evidence_key=evidence_key,
                meta_key=meta_key,
                include_evidence=include_evidence,
                include_meta=include_meta,
            )
            if context_parts:
                user_lines.append("Контекст:")
                user_lines.extend(context_parts)

            user_message = "\n".join([line for line in user_lines if line])

            entry = {
                "request": [
                    {"role": "system", "text": system_message},
                    {"role": "user", "text": user_message},
                ],
                "response": response_str,
            }
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")
            total_written += 1

            if limit and total_written >= limit:
                break

    return total_written


def _parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвертация маркетингового JSONL в формат чата Яндекса."
    )
    parser.add_argument("input", type=Path, help="Входной JSONL в нашем внутреннем формате.")
    parser.add_argument("output", type=Path, help="Целевой JSONL в формате Яндекса.")
    parser.add_argument("--split", help="Фильтр по split (train/test/all).")
    parser.add_argument("--limit", type=int, help="Ограничить количество сконвертированных записей.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Системное сообщение (request[0]).",
    )
    parser.add_argument("--include-evidence", action="store_true", help="Добавлять evidence в текст пользователя.")
    parser.add_argument("--include-meta", action="store_true", help="Добавлять meta в текст пользователя.")
    parser.add_argument("--include-labels", action="store_true", help="Добавлять список labels в текст пользователя.")
    parser.add_argument("--prompt-key", default="prompt", help="Ключ с вопросом/инструкцией.")
    parser.add_argument("--response-key", default="response", help="Ключ с ожидаемым ответом.")
    parser.add_argument("--task-key", default="task", help="Ключ с типом задания.")
    parser.add_argument("--evidence-key", default="evidence", help="Ключ с контекстом/evidence.")
    parser.add_argument("--labels-key", default="labels", help="Ключ со списком меток.")
    parser.add_argument("--meta-key", default="meta", help="Ключ с метаданными.")
    parser.add_argument("--split-key", default="split", help="Ключ с информацией о split.")
    return parser.parse_args(list(argv) if argv is not None else None)


def _main_cli(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_cli_args(argv)
    try:
        written = convert_to_yandex_format(
            args.input,
            args.output,
            split=args.split,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            task_key=args.task_key,
            evidence_key=args.evidence_key,
            labels_key=args.labels_key,
            meta_key=args.meta_key,
            split_key=args.split_key,
            system_prompt=args.system_prompt,
            include_evidence=args.include_evidence,
            include_meta=args.include_meta,
            include_labels=args.include_labels,
            limit=args.limit,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[error] Ошибка во время конвертации: {exc}")
        return 1
    print(f"[info] Конвертировано {written} примеров в '{args.output}'.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main_cli())


__all__ = [
    "load_examples_from_jsonl",
    "clean_value",
    "convert_to_yandex_format",
    "DEFAULT_SYSTEM_PROMPT",
]
