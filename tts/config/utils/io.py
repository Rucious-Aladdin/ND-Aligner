import json
from dataclasses import Field, asdict
from typing import Any, ClassVar, Protocol, TypeVar

from dacite import from_dict


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


T = TypeVar("T")


def save_config(
    cfg: DataclassInstance,
    save_path: str = "config.json",
) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4, ensure_ascii=False)


def load_config(
    cfg_path: str,
    cls_type: type[T],
) -> T:
    """
    Reads a JSON configuration file and restores it into a specified dataclass object.

    By using a generic type (TypeVar 'T'), the return type automatically matches
    the class type passed during the function call. This ensures that IDEs (like VSCode)
    can provide perfect autocompletion and type checking for the returned object's attributes.

    Args:
        cfg_path (str): The path to the JSON configuration file to read.
        cls_type (type[T]): The target dataclass 'type' itself.
                            (**The class name, not an instance.** e.g., DataConfig)

    Returns:
        T: An instance of the specified dataclass populated with the JSON data.

    Example:
        >>> # Assuming a dataclass named 'DataConfig' is defined
        >>> loaded_cfg = load_config("config_v1.json", DataConfig)
        >>>
        >>> # The IDE recognizes loaded_cfg as a DataConfig type, enabling autocompletion
        >>> print(loaded_cfg.train.batch_size)
        16
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # Convert the dictionary back into a dataclass using the dacite library
    loaded_config = from_dict(data_class=cls_type, data=config_dict)

    return loaded_config


def print_config(cfg: DataclassInstance) -> None:
    print("=" * 50)
    print("[ Current Configuration ]")
    print("-" * 50)

    def _print_node(data: Any, indent_level: int = 0):
        indent_str = "    " * indent_level

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{indent_str}{key}:")
                    _print_node(value, indent_level + 1)
                else:
                    print(f"{indent_str}{key}: {value}")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    print(f"{indent_str}-")
                    _print_node(item, indent_level + 1)
                else:
                    print(f"{indent_str}- {item}")
        else:
            print(f"{indent_str}{data}")

    _print_node(asdict(cfg))
    print("=" * 50)
