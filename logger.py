"""
logger.py
============================
统一日志管理模块，所有日志统一存放在 logs/ 文件夹中。
"""

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name: str = "knowledge_base") -> logging.Logger:
    """
    设置日志记录器，日志输出到 logs/ 文件夹。
    
    Args:
        name: 日志记录器名称
    
    Returns:
        Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_exception(logger: logging.Logger, message: str, exc: Exception = None):
    """
    记录异常信息。
    
    Args:
        logger: 日志记录器
        message: 错误消息
        exc: 异常对象
    """
    if exc:
        logger.error(f"{message}: {exc}", exc_info=True)
    else:
        logger.error(message)

def log_performance(logger: logging.Logger, operation: str, duration: float, file_name: str = ""):
    """
    记录性能指标。
    
    Args:
        logger: 日志记录器
        operation: 操作名称
        duration: 耗时（秒）
        file_name: 处理的文件名
    """
    logger.info(f"[PERF] {operation} completed in {duration:.2f}s" + (f" for {file_name}" if file_name else ""))

def get_log_file_path(name: str = "knowledge_base") -> str:
    """
    获取当前日志文件路径。
    
    Args:
        name: 日志记录器名称
    
    Returns:
        日志文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    return os.path.join(LOG_DIR, f"{name}_{timestamp}.log")

logger = setup_logger()
