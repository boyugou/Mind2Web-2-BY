import logging
import sys
import os
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pythonjsonlogger import jsonlogger


class ColoredStructuredFormatter(logging.Formatter):
    """Colored structured log formatter"""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'RESET': '\033[0m'
    }

    def format(self, record):
        # For verification operations, use special format
        if hasattr(record, 'op_id'):
            op_id = record.op_id
            level_color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']

            # Build main message
            msg_parts = [
                f"{level_color}[{record.levelname}]{reset}",
                f"[{op_id}]"
            ]

            # Add node information
            if hasattr(record, 'id') and record.node_id:
                msg_parts.append(f"Node({record.node_id})")

            # Add verification type
            if hasattr(record, 'verify_type'):
                msg_parts.append(f"<{record.verify_type}>")

            # Add main message
            msg_parts.append(record.getMessage())

            # Build detailed information (indented display)
            details = []

            if hasattr(record, 'node_desc') and record.node_desc:
                details.append(f"  📋 Description: {record.node_desc}")

            if hasattr(record, 'url') and record.url:
                details.append(f"  🔗 URL: {record.url}")

            if hasattr(record, 'claim_preview'):
                details.append(f"  💬 Claim: {record.claim_preview}")

            if hasattr(record, 'reasoning') and record.reasoning:
                reasoning = record.reasoning
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
                details.append(f"  💭 Reasoning: {reasoning}")

            if hasattr(record, 'result'):
                result_str = "✅ PASS" if record.result else "❌ FAIL"
                details.append(f"  📊 Result: {result_str}")

            # Combine all parts
            full_msg = " ".join(msg_parts)
            if details:
                full_msg += "\n" + "\n".join(details)

            return full_msg

        # For other logs, use standard format
        level_color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        return f"{level_color}[{record.levelname}]{reset} {record.getMessage()}"


def create_logger(lgr_nm: str, log_folder: str, enable_console: bool = True) -> tuple[Logger, str]:
    """
    Create independent logger instance, avoid handler sharing issues

    Args:
        lgr_nm: logger name
        log_folder: log folder
        enable_console: whether to enable console output

    Returns:
        (logger instance, timestamp)
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{current_time}_{lgr_nm}.jsonl"
    log_file_path = os.path.join(log_folder, log_file_name)

    # Create unique logger name, avoid repetition
    unique_logger_name = f"{lgr_nm}_{current_time}_{id(log_folder)}"

    # Check if logger already exists, if so, clean up first
    existing_logger = logging.getLogger(unique_logger_name)
    if existing_logger.handlers:
        for handler in existing_logger.handlers[:]:
            existing_logger.removeHandler(handler)
            handler.close()

    # Create new logger
    new_logger = logging.getLogger(unique_logger_name)
    new_logger.setLevel(logging.DEBUG)

    # Prevent log propagation to root logger
    new_logger.propagate = False

    # File handler - Use JSON format for machine processing
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when="D",
        backupCount=14,
        encoding="utf-8"
    )
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(logging.DEBUG)
    new_logger.addHandler(file_handler)

    # Console handler - Use colored structured format for human reading
    if enable_console:
        console_handler = StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredStructuredFormatter())
        console_handler.setLevel(logging.INFO)
        new_logger.addHandler(console_handler)

    return new_logger, current_time


def create_sub_logger(parent_logger: Logger, sub_name: str) -> Logger:
    """
    Create sublogger based on parent logger, inherit parent logger's handlers
    Used to create hierarchical logs within the same evaluation
    """
    parent_name = parent_logger.name
    sub_logger_name = f"{parent_name}.{sub_name}"

    sub_logger = logging.getLogger(sub_logger_name)
    sub_logger.setLevel(parent_logger.level)
    sub_logger.propagate = True  # Allow propagation to parent logger

    return sub_logger


def cleanup_logger(logger: Logger) -> None:
    """Clean up all handlers of logger"""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()