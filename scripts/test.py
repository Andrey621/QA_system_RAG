#!/usr/bin/env python
import json
import difflib
import requests

DATASET = "app/data_for_train/lat_marketing_dataset.jsonl"
BASE_URL = "http://localhost:8000"
ENDPOINT = "/compare_answers/"
USER_ID = 42
LIMIT = 5


def _normalize(text: str) -> str:
    return text.strip().lower()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    with open(DATASET, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()][:LIMIT]

    url = f"{BASE_URL.rstrip('/')}{ENDPOINT}"
    total = len(lines)
    local_hits = 0
    yandex_hits = 0
    local_sim_sum = 0.0
    yandex_sim_sum = 0.0
    pair_sim_sum = 0.0
    quality_scores = []
    cosine_scores = []

    for idx, sample in enumerate(lines, 1):
        question = sample["request"][0]['text']
        answer = sample["response"]

        payload = {"user_id": USER_ID, "question": question}
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        expected = answer
        local_answer = data.get("local_answer", "")
        yandex_answer = data.get("yandex_answer", "")

        local_ok = _normalize(local_answer) == _normalize(expected)
        yandex_ok = _normalize(yandex_answer) == _normalize(expected)
        if local_ok:
            local_hits += 1
        if yandex_ok:
            yandex_hits += 1

        local_sim = _similarity(local_answer, expected)
        yandex_sim = _similarity(yandex_answer, expected)
        pair_sim = _similarity(local_answer, yandex_answer) if local_answer and yandex_answer else 0.0
        local_sim_sum += local_sim
        yandex_sim_sum += yandex_sim
        pair_sim_sum += pair_sim

        quality = data.get("quality_score")
        cosine = data.get("cosine_similarity_score")
        if isinstance(quality, (int, float)):
            quality_scores.append(quality)
        if isinstance(cosine, (int, float)):
            cosine_scores.append(cosine)

        print(f"#{idx}")
        print(f"  ожидание : {expected}")
        print(f"  local ({'OK' if local_ok else 'MISS'} | sim={local_sim*100:.1f}%): {local_answer}")
        print(f"  yandex({'OK' if yandex_ok else 'MISS'} | sim={yandex_sim*100:.1f}%): {yandex_answer}")
        if quality is not None:
            print(f"  local <-> yandex quality: {quality:.2f}")
        if cosine is not None:
            print(f"  local <-> yandex cosine: {cosine:.3f}")
        print(f"  local <-> yandex text similarity: {pair_sim*100:.1f}%")
        print("-" * 40)

    if total:
        print(f"Local accuracy: {local_hits}/{total} = {local_hits/total:.2%}")
        print(f"Yandex accuracy: {yandex_hits}/{total} = {yandex_hits/total:.2%}")
        print(f"Avg local textual similarity: {local_sim_sum/total:.2%}")
        print(f"Avg yandex textual similarity: {yandex_sim_sum/total:.2%}")
        print(f"Avg local <-> yandex textual similarity: {pair_sim_sum/total:.2%}")
        if quality_scores:
            print(f"Avg local <-> yandex quality score: {sum(quality_scores)/len(quality_scores):.2f}")
        if cosine_scores:
            print(f"Avg local <-> yandex cosine: {sum(cosine_scores)/len(cosine_scores):.3f}")
    else:
        print("Нет примеров для оценки.")


if __name__ == "__main__":
    main()
