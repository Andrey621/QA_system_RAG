"""
Простой менеджер фоновых задач для запросов на дообучение

Использует пул потоков для выполнения блокирующих задач дообучения без
занимания цикла запросов FastAPI. Отслеживает статус каждой задачи в памяти
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, Optional

from fine_tuning import fine_tune_local_model
from models import FineTuneRequest


class TrainingJobManager:
    """Координирует выполнение и отслеживание статуса задач дообучения"""

    def __init__(self, *, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fine-tune")
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Optional[str]]] = {}

    def submit_job(
        self,
        request_data: FineTuneRequest,
        *,
        default_model_name: str,
        default_output_dir: str,
    ) -> str:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "message": "Задача дообучения добавлена в очередь",
                "model_path": None,
            }

        def _run_job():
            self._update_job(job_id, status="running", message="Задача дообучения запущена.")
            base_model = request_data.base_model_name or default_model_name
            output_dir = request_data.output_dir or default_output_dir
            try:
                model_path = fine_tune_local_model(
                    examples=request_data.examples,
                    base_model_name=base_model,
                    output_dir=output_dir,
                    run_name=request_data.run_name,
                    learning_rate=request_data.learning_rate,
                    num_epochs=request_data.num_epochs,
                    max_seq_length=request_data.max_seq_length,
                    gradient_accumulation_steps=request_data.gradient_accumulation_steps,
                    trust_remote_code=request_data.trust_remote_code,
                )
            except Exception as exc:  # pylint: disable=broad-except
                self._update_job(
                    job_id,
                    status="failed",
                    message=f"Ошибка дообучения: {exc}",
                    model_path=None,
                )
                return
            self._update_job(
                job_id,
                status="completed",
                message=f"Модель сохранена: {model_path}",
                model_path=model_path,
            )

        self._executor.submit(_run_job)
        return job_id

    def _update_job(
        self,
        job_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = status
            if message is not None:
                job["message"] = message
            if model_path is not None:
                job["model_path"] = model_path

    def get_status(self, job_id: str) -> Optional[Dict[str, Optional[str]]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return deepcopy(job)
