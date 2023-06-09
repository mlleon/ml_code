import logging
from logging.handlers import RotatingFileHandler


class NoNewlineFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return msg.replace('\n', ' ')


# 配置日志记录
def configure_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理器，将日志信息写入到指定文件中
    file_handler = RotatingFileHandler(log_file, maxBytes=1048576, backupCount=2)  # 指定文件大小和备份个数
    file_handler.setLevel(logging.INFO)

    # 创建日志格式器
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # 将格式器添加到文件处理器
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到日志记录器
    logger.addHandler(file_handler)

    return logger
