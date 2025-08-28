import logging as _base_logging
import inspect

class ColoredFormatter(_base_logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",   
        "INFO": "\033[36m",   
        "WARNING": "\033[33m", 
        "ERROR": "\033[31m",   
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

class Logger:
    def __init__(self, name: str = __name__):
        self.logger = _base_logging.getLogger(name)
        handler = _base_logging.StreamHandler()
        formatter = ColoredFormatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(_base_logging.DEBUG)

    def _get_context_tag(self):
        stack = inspect.stack()
        for frame_info in stack:
            func_name = frame_info.function
            if func_name not in {"info", "debug", "warning", "error", "_get_context_tag"}:
                class_name = None
                locals_ = frame_info.frame.f_locals
                if "self" in locals_:
                    class_name = locals_["self"].__class__.__name__
                elif "cls" in locals_:
                    class_name = locals_["cls"].__name__
                return f"[{class_name or 'NoClass'}[{func_name}]]"
        return "[Unknown]"

    def info(self, message: str, *args, **kwargs):
        self.logger.info(f"{self._get_context_tag()} {message}", *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(f"{self._get_context_tag()} {message}", *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(f"{self._get_context_tag()} {message}", *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self.logger.error(f"{self._get_context_tag()} {message}", *args, **kwargs)

logger = Logger(__name__)






if __name__ == "__main__":
    import asyncio
    loggertest = Logger(__name__)

    class Service:
        def sync_method(self):
            loggertest.info("Sync method running")

        async def async_method(self):
            await asyncio.sleep(0.1)
            loggertest.info("Async method running")

    service = Service()

    service.sync_method()
    asyncio.run(service.async_method())