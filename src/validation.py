"""
Validation Module for PixelNur Phase 2
Provides comprehensive input validation for all system components

Requirements:
- 17.1: Validate unsupported image formats
- 17.2: Validate message length against capacity
- 17.5: Validate image dimensions (minimum 256×256)
- 17.6: Validate image dimensions (maximum 7680×4320 for 8K)
- 17.7: Validate message length does not exceed embedding capacity
- 17.8: Validate encryption keys are at least 16 characters
- 21.1-21.5: Support PNG, JPEG, BMP, TIFF, WebP formats
"""

import os
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class ImageFormatError(ValidationError):
    """Exception raised for invalid image formats."""
    pass


class ImageDimensionError(ValidationError):
    """Exception raised for invalid image dimensions."""
    pass


class MessageLengthError(ValidationError):
    """Exception raised for invalid message lengths."""
    pass


class KeyValidationError(ValidationError):
    """Exception raised for invalid encryption keys."""
    pass


class ParameterValidationError(ValidationError):
    """Exception raised for invalid parameters."""
    pass


# Supported image formats (Requirements 21.1-21.5)
SUPPORTED_FORMATS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.bmp': 'BMP',
    '.tiff': 'TIFF',
    '.tif': 'TIFF',
    '.webp': 'WebP'
}

# Magic bytes for format validation
MAGIC_BYTES = {
    'PNG': b'\x89PNG\r\n\x1a\n',
    'JPEG': b'\xff\xd8\xff',
    'BMP': b'BM',
    'TIFF_LE': b'II\x2a\x00',  # Little-endian TIFF
    'TIFF_BE': b'MM\x00\x2a',  # Big-endian TIFF
    'WEBP': b'RIFF'
}

# Image dimension constraints (Requirements 17.5, 17.6)
MIN_IMAGE_WIDTH = 256
MIN_IMAGE_HEIGHT = 256
MAX_IMAGE_WIDTH = 7680  # 8K width
MAX_IMAGE_HEIGHT = 4320  # 8K height

# Encryption key constraints (Requirement 17.8)
MIN_KEY_LENGTH = 16

# Robustness level options
VALID_ROBUSTNESS_LEVELS = ['none', 'low', 'medium', 'high']

# Embedding strength constraints
MIN_EMBEDDING_STRENGTH = 0.5
MAX_EMBEDDING_STRENGTH = 2.0


def validate_image_format_by_extension(file_path: str) -> str:
    """
    Validate image format by file extension.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Format name (PNG, JPEG, BMP, TIFF, WebP)
        
    Raises:
        ImageFormatError: If format is not supported
        
    Requirements: 17.1, 21.1-21.5
    """
    if not isinstance(file_path, str):
        raise TypeError("File path must be a string")
    
    if not file_path:
        raise ImageFormatError("File path cannot be empty")
    
    # Extract extension
    _, ext = os.path.splitext(file_path.lower())
    
    if ext not in SUPPORTED_FORMATS:
        supported_list = ', '.join(sorted(set(SUPPORTED_FORMATS.values())))
        raise ImageFormatError(
            f"Unsupported image format: '{ext}'. "
            f"Supported formats: {supported_list} "
            f"(extensions: {', '.join(sorted(SUPPORTED_FORMATS.keys()))})"
        )
    
    return SUPPORTED_FORMATS[ext]


def validate_image_format_by_magic_bytes(file_path: str) -> str:
    """
    Validate image format by checking magic bytes (file signature).
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Format name (PNG, JPEG, BMP, TIFF, WebP)
        
    Raises:
        ImageFormatError: If format cannot be determined or is not supported
        FileNotFoundError: If file does not exist
        
    Requirements: 17.1, 21.1-21.5
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)  # Read first 12 bytes
    except Exception as e:
        raise ImageFormatError(f"Failed to read image file: {e}")
    
    if not header:
        raise ImageFormatError("Image file is empty")
    
    # Check magic bytes
    if header.startswith(MAGIC_BYTES['PNG']):
        return 'PNG'
    elif header.startswith(MAGIC_BYTES['JPEG']):
        return 'JPEG'
    elif header.startswith(MAGIC_BYTES['BMP']):
        return 'BMP'
    elif header.startswith(MAGIC_BYTES['TIFF_LE']) or header.startswith(MAGIC_BYTES['TIFF_BE']):
        return 'TIFF'
    elif header.startswith(MAGIC_BYTES['WEBP']):
        return 'WebP'
    else:
        raise ImageFormatError(
            f"Unrecognized image format. File does not match any supported magic bytes. "
            f"Supported formats: PNG, JPEG, BMP, TIFF, WebP"
        )


def validate_image_format(file_path: str, check_magic_bytes: bool = True) -> str:
    """
    Validate image format by both extension and magic bytes.
    
    Args:
        file_path: Path to the image file
        check_magic_bytes: Whether to verify magic bytes (default: True)
        
    Returns:
        Format name (PNG, JPEG, BMP, TIFF, WebP)
        
    Raises:
        ImageFormatError: If format is invalid or mismatched
        
    Requirements: 17.1, 21.1-21.5
    """
    # Validate extension
    format_by_ext = validate_image_format_by_extension(file_path)
    
    # Optionally validate magic bytes
    if check_magic_bytes and os.path.exists(file_path):
        format_by_magic = validate_image_format_by_magic_bytes(file_path)
        
        # Warn if mismatch (but don't fail - trust magic bytes)
        if format_by_ext != format_by_magic:
            # Note: In production, you might want to log this warning
            pass
        
        return format_by_magic
    
    return format_by_ext


def validate_image_dimensions(
    image: Union[np.ndarray, Image.Image, Tuple[int, int]],
    min_width: int = MIN_IMAGE_WIDTH,
    min_height: int = MIN_IMAGE_HEIGHT,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT
) -> Tuple[int, int]:
    """
    Validate image dimensions are within acceptable range.
    
    Args:
        image: Image as numpy array, PIL Image, or (width, height) tuple
        min_width: Minimum width in pixels (default: 256)
        min_height: Minimum height in pixels (default: 256)
        max_width: Maximum width in pixels (default: 7680 for 8K)
        max_height: Maximum height in pixels (default: 4320 for 8K)
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        ImageDimensionError: If dimensions are invalid
        
    Requirements: 17.5, 17.6
    """
    # Extract dimensions based on input type
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ImageDimensionError("Image array is empty")
        if len(image.shape) < 2:
            raise ImageDimensionError(f"Image must be at least 2D, got shape: {image.shape}")
        height, width = image.shape[:2]
    elif isinstance(image, Image.Image):
        width, height = image.size
    elif isinstance(image, tuple) and len(image) == 2:
        width, height = image
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("Dimensions must be integers")
    else:
        raise TypeError(
            "Image must be numpy array, PIL Image, or (width, height) tuple"
        )
    
    # Validate minimum dimensions (Requirement 17.5)
    if width < min_width or height < min_height:
        raise ImageDimensionError(
            f"Image too small: minimum {min_width}×{min_height} pixels required, "
            f"got {width}×{height} pixels. "
            f"Suggestion: Use a larger image or resize to at least {min_width}×{min_height}."
        )
    
    # Validate maximum dimensions (Requirement 17.6)
    if width > max_width or height > max_height:
        raise ImageDimensionError(
            f"Image too large: maximum {max_width}×{max_height} pixels (8K), "
            f"got {width}×{height} pixels. "
            f"Suggestion: Resize image to {max_width}×{max_height} or smaller."
        )
    
    return width, height


def validate_message_length(
    message: Union[str, bytes],
    max_length: Optional[int] = None,
    capacity: Optional[int] = None
) -> int:
    """
    Validate message length against constraints.
    
    Args:
        message: Message to validate (string or bytes)
        max_length: Maximum allowed length in bytes (optional)
        capacity: Available embedding capacity in bytes (optional)
        
    Returns:
        Message length in bytes
        
    Raises:
        MessageLengthError: If message is too long
        
    Requirements: 17.2, 17.7
    """
    if message is None:
        raise MessageLengthError("Message cannot be None")
    
    # Convert to bytes if string
    if isinstance(message, str):
        message_bytes = message.encode('utf-8')
    elif isinstance(message, bytes):
        message_bytes = message
    else:
        raise TypeError("Message must be string or bytes")
    
    message_length = len(message_bytes)
    
    if message_length == 0:
        raise MessageLengthError("Message cannot be empty")
    
    # Check against maximum length
    if max_length is not None and message_length > max_length:
        raise MessageLengthError(
            f"Message too long: {message_length} bytes exceeds maximum of {max_length} bytes"
        )
    
    # Check against capacity (Requirement 17.7)
    if capacity is not None and message_length > capacity:
        raise MessageLengthError(
            f"Message too long: {message_length} bytes exceeds embedding capacity of {capacity} bytes. "
            f"Suggestions:\n"
            f"  - Reduce message length to {capacity} bytes or less\n"
            f"  - Use a larger cover image\n"
            f"  - Reduce robustness level (if enabled) to increase capacity"
        )
    
    return message_length


def validate_encryption_key(key: str, min_length: int = MIN_KEY_LENGTH) -> None:
    """
    Validate encryption key meets security requirements.
    
    Args:
        key: Encryption key to validate
        min_length: Minimum required key length (default: 16)
        
    Raises:
        KeyValidationError: If key is invalid
        
    Requirements: 17.8
    """
    if not isinstance(key, str):
        raise TypeError("Encryption key must be a string")
    
    if not key:
        raise KeyValidationError("Encryption key cannot be empty")
    
    if len(key) < min_length:
        raise KeyValidationError(
            f"Encryption key too short: minimum {min_length} characters required, "
            f"got {len(key)} characters. "
            f"Suggestion: Use a longer passphrase or add more characters."
        )


def validate_encryption_key_strength(key: str) -> dict:
    """
    Assess encryption key strength and provide recommendations.
    
    Args:
        key: Encryption key to assess
        
    Returns:
        Dictionary with strength assessment:
        - 'length': Key length
        - 'has_uppercase': Contains uppercase letters
        - 'has_lowercase': Contains lowercase letters
        - 'has_digits': Contains digits
        - 'has_special': Contains special characters
        - 'strength': 'weak', 'medium', 'strong'
        - 'recommendations': List of improvement suggestions
        
    Requirements: 17.8
    """
    validate_encryption_key(key)  # First validate minimum requirements
    
    assessment = {
        'length': len(key),
        'has_uppercase': any(c.isupper() for c in key),
        'has_lowercase': any(c.islower() for c in key),
        'has_digits': any(c.isdigit() for c in key),
        'has_special': any(not c.isalnum() for c in key),
        'recommendations': []
    }
    
    # Calculate strength score
    score = 0
    if assessment['length'] >= 16:
        score += 1
    if assessment['length'] >= 24:
        score += 1
    if assessment['has_uppercase']:
        score += 1
    if assessment['has_lowercase']:
        score += 1
    if assessment['has_digits']:
        score += 1
    if assessment['has_special']:
        score += 1
    
    # Determine strength level
    if score <= 2:
        assessment['strength'] = 'weak'
    elif score <= 4:
        assessment['strength'] = 'medium'
    else:
        assessment['strength'] = 'strong'
    
    # Generate recommendations
    if assessment['length'] < 24:
        assessment['recommendations'].append("Use at least 24 characters for stronger security")
    if not assessment['has_uppercase']:
        assessment['recommendations'].append("Include uppercase letters")
    if not assessment['has_lowercase']:
        assessment['recommendations'].append("Include lowercase letters")
    if not assessment['has_digits']:
        assessment['recommendations'].append("Include digits")
    if not assessment['has_special']:
        assessment['recommendations'].append("Include special characters")
    
    return assessment


def validate_robustness_level(level: str) -> str:
    """
    Validate robustness level parameter.
    
    Args:
        level: Robustness level ('none', 'low', 'medium', 'high')
        
    Returns:
        Validated robustness level (lowercase)
        
    Raises:
        ParameterValidationError: If level is invalid
    """
    if not isinstance(level, str):
        raise TypeError("Robustness level must be a string")
    
    level_lower = level.lower()
    
    if level_lower not in VALID_ROBUSTNESS_LEVELS:
        raise ParameterValidationError(
            f"Invalid robustness level: '{level}'. "
            f"Valid options: {', '.join(VALID_ROBUSTNESS_LEVELS)}"
        )
    
    return level_lower


def validate_embedding_strength(strength: float) -> float:
    """
    Validate embedding strength parameter.
    
    Args:
        strength: Embedding strength multiplier (0.5 to 2.0)
        
    Returns:
        Validated embedding strength
        
    Raises:
        ParameterValidationError: If strength is out of range
    """
    if not isinstance(strength, (int, float)):
        raise TypeError("Embedding strength must be a number")
    
    if strength < MIN_EMBEDDING_STRENGTH or strength > MAX_EMBEDDING_STRENGTH:
        raise ParameterValidationError(
            f"Embedding strength out of range: must be between "
            f"{MIN_EMBEDDING_STRENGTH} and {MAX_EMBEDDING_STRENGTH}, got {strength}. "
            f"Lower values = more imperceptible, higher values = more robust."
        )
    
    return float(strength)


def validate_image_array(image: np.ndarray, name: str = "image") -> None:
    """
    Validate numpy image array format and properties.
    
    Args:
        image: Image array to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If image array is invalid
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    
    if image.size == 0:
        raise ValidationError(f"{name} array is empty")
    
    if len(image.shape) not in [2, 3]:
        raise ValidationError(
            f"{name} must be 2D (grayscale) or 3D (RGB), got shape: {image.shape}"
        )
    
    # If RGB, must have 3 channels
    if len(image.shape) == 3 and image.shape[2] != 3:
        raise ValidationError(
            f"{name} must have 3 channels (RGB), got {image.shape[2]} channels"
        )
    
    # Validate dimensions
    validate_image_dimensions(image)


def validate_all_embedding_inputs(
    cover_image: np.ndarray,
    message: Union[str, bytes],
    key: str,
    capacity: int,
    robustness_level: str = 'none',
    embedding_strength: float = 1.0
) -> dict:
    """
    Validate all inputs for embedding operation.
    
    Args:
        cover_image: Cover image array
        message: Message to embed
        key: Encryption key
        capacity: Available embedding capacity in bytes
        robustness_level: Robustness level setting
        embedding_strength: Embedding strength multiplier
        
    Returns:
        Dictionary with validated parameters
        
    Raises:
        ValidationError: If any validation fails
    """
    # Validate cover image
    validate_image_array(cover_image, "Cover image")
    width, height = validate_image_dimensions(cover_image)
    
    # Validate message
    message_length = validate_message_length(message, capacity=capacity)
    
    # Validate encryption key
    validate_encryption_key(key)
    
    # Validate robustness level
    validated_level = validate_robustness_level(robustness_level)
    
    # Validate embedding strength
    validated_strength = validate_embedding_strength(embedding_strength)
    
    return {
        'image_dimensions': (width, height),
        'message_length': message_length,
        'robustness_level': validated_level,
        'embedding_strength': validated_strength,
        'capacity_used': message_length,
        'capacity_available': capacity,
        'capacity_utilization': (message_length / capacity * 100) if capacity > 0 else 0
    }


def validate_all_extraction_inputs(
    stego_image: np.ndarray,
    key: str
) -> dict:
    """
    Validate all inputs for extraction operation.
    
    Args:
        stego_image: Stego image array
        key: Encryption key
        
    Returns:
        Dictionary with validated parameters
        
    Raises:
        ValidationError: If any validation fails
    """
    # Validate stego image
    validate_image_array(stego_image, "Stego image")
    width, height = validate_image_dimensions(stego_image)
    
    # Validate encryption key
    validate_encryption_key(key)
    
    return {
        'image_dimensions': (width, height)
    }
