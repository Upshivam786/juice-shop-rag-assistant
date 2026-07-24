"""
Structured logging (Point 10 - Logging & Observability).

Why JSON logs: plain print()/f-string logs (what the original code used for
"STEP 1/2/3/4" debugging) are fine for local debugging but useless for
production - they can't be filtered, aggregated, or queried by a log platform
(CloudWatch, Loki, Datadog, etc). Structured (JSON) logs with consistent field
names let you answer questions like "p95 retrieval latency in the last hour"
or "which questions triggered LLM errors" with a log query instead of grepping
raw text.

This module gives every log line a consistent shape:
  {"timestamp": ..., "level": ..., "event": ..., ...extra fields}
"""
import json
import logging
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Any extra=... fields passed to logger calls get merged in.
        for key, value in record.__dict__.items():
            if key in (
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message",
            ):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
    root.addHandler(handler)

    # Quiet down noisy third-party loggers unless we're debugging.
    if level.upper() != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
