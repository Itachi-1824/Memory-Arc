"""Configuration file loading utilities for InfiniteContextEngine."""

import json
import logging
from pathlib import Path
from typing import Any

from .models import InfiniteContextConfig

logger = logging.getLogger(__name__)


def load_config_from_file(file_path: str | Path) -> InfiniteContextConfig:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        file_path: Path to configuration file (.json or .yaml/.yml)
        
    Returns:
        InfiniteContextConfig instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".json":
        return _load_json_config(file_path)
    elif suffix in [".yaml", ".yml"]:
        return _load_yaml_config(file_path)
    else:
        raise ValueError(
            f"Unsupported configuration file format: {suffix}. "
            "Supported formats: .json, .yaml, .yml"
        )


def _load_json_config(file_path: Path) -> InfiniteContextConfig:
    """Load configuration from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        config = InfiniteContextConfig.from_dict(config_dict)
        config.validate()
        
        logger.info(f"Loaded configuration from {file_path}")
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def _load_yaml_config(file_path: Path) -> InfiniteContextConfig:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load YAML configuration files. "
            "Install it with: pip install pyyaml"
        )
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        if not isinstance(config_dict, dict):
            raise ValueError("YAML file must contain a dictionary")
        
        config = InfiniteContextConfig.from_dict(config_dict)
        config.validate()
        
        logger.info(f"Loaded configuration from {file_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def save_config_to_file(
    config: InfiniteContextConfig,
    file_path: str | Path,
    format: str = "json"
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        file_path: Path to save configuration
        format: File format ('json' or 'yaml')
        
    Raises:
        ValueError: If format is unsupported
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    if format == "json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved configuration to {file_path}")
        
    elif format == "yaml":
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to save YAML configuration files. "
                "Install it with: pip install pyyaml"
            )
        
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        logger.info(f"Saved configuration to {file_path}")
        
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
