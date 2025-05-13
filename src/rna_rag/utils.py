import os
from typing import Optional

def batch_list(lst: list, batch_size: int) -> list:
    """Split a list into batches of specified size."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def get_required_env_var(name: str) -> str:
    """
    Get a required environment variable or raise an error if not set.
    
    Args:
        name: Name of the environment variable to get
        
    Returns:
        The value of the environment variable
        
    Raises:
        ValueError: If the environment variable is not set
    """
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Required environment variable {name} is not set")
    return value

