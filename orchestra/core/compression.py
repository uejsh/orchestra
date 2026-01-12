# compression.py - Storage optimization
# ============================================================================
# FILE: orchestra/core/compression.py
# Compression: zlib (Level 1) + Summarization (Level 2 - Optional)
# ============================================================================

import zlib
import pickle
import logging
from typing import Any, Optional, Callable, Union

logger = logging.getLogger(__name__)

class CompressionManager:
    """
    Manages data compression to reduce memory/disk usage.
    """
    
    def __init__(self, enable_compression: bool = True, compression_level: int = 6):
        self.enabled = enable_compression
        self.level = compression_level
    
    def compress(self, value: Any) -> Union[bytes, Any]:
        """Compress a value if enabled"""
        if not self.enabled:
            return value
            
        try:
            # 1. Serialize
            data = pickle.dumps(value)
            
            # 2. Compress
            compressed = zlib.compress(data, level=self.level)
            
            ratio = len(compressed) / len(data)
            logger.debug(f"Compressed {len(data)}b -> {len(compressed)}b (Ratio: {ratio:.2f})")
            
            return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {e}. Storing uncompressed.")
            return value

    def decompress(self, data: Union[bytes, Any]) -> Any:
        """Decompress a value if it appears compressed"""
        if not self.enabled:
            return data
            
        if not isinstance(data, bytes):
            return data
        
        try:
            # Check for zlib signature roughly or just try
            try:
                decompressed = zlib.decompress(data)
                return pickle.loads(decompressed)
            except zlib.error:
                # Wasn't compressed data
                return data
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data

    def size_of(self, value: Any) -> int:
        """Get approximate size in bytes"""
        if isinstance(value, bytes):
            return len(value)
        try:
            return len(pickle.dumps(value))
        except:
            return 0
