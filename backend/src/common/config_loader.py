import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Tự động tính toán đường dẫn tới thư mục gốc (Root Directory)
# backend/src/common/config_loader.py -> lên 4 cấp thư mục là Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"
ENV_PATH = BASE_DIR / ".env"

# Nạp các biến môi trường từ file .env vào os.environ
load_dotenv(dotenv_path=ENV_PATH)

# Regex pattern để bắt cú pháp ${VAR_NAME:-default_value} hoặc ${VAR_NAME}
ENV_PATTERN = re.compile(r'\$\{([^}^{]+)\}')

def _resolve_env_var(match) -> str:
    """Hàm nội bộ để giải mã regex biến môi trường."""
    env_var = match.group(1)
    default_value = ""
    
    if ':-' in env_var:
        env_var, default_value = env_var.split(':-', 1)
        
    return os.environ.get(env_var, default_value)

def _env_constructor(loader, node) -> Any:
    """Constructor tùy chỉnh cho thư viện PyYAML."""
    value = loader.construct_scalar(node)
    return ENV_PATTERN.sub(_resolve_env_var, value)

# Gắn constructor vào PyYAML để tự động kích hoạt khi gặp chuỗi
yaml.SafeLoader.add_implicit_resolver('!env', ENV_PATTERN, None)
yaml.SafeLoader.add_constructor('!env', _env_constructor)
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:str', _env_constructor)

class ConfigLoader:
    _instance: Dict[str, Any] = None

    @classmethod
    def get_config(cls, config_path: str = None) -> Dict[str, Any]:
        """Load cấu hình một lần duy nhất (Singleton) và trả về dictionary."""
        if cls._instance is None:
            path_to_open = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
            
            if not path_to_open.exists():
                raise FileNotFoundError(f"[ConfigLoader] Không tìm thấy file cấu hình tại {path_to_open}")
                
            with open(path_to_open, 'r', encoding='utf-8') as f:
                cls._instance = yaml.safe_load(f)
                
            print(f"[ConfigLoader] Đã nạp thành công cấu hình từ {path_to_open}")
            
        return cls._instance

# Biến toàn cục để các module khác import trực tiếp
# Cú pháp sử dụng ở file khác: 
# from backend.src.common.config_loader import config
config = ConfigLoader.get_config()