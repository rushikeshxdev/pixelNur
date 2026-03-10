# EncryptionService Documentation

## Overview

The `EncryptionService` class provides Slice Encryption functionality using SHA-256 + XOR for the PixelNur Phase 2 steganography system.

## Features

- **SHA-256 Hash Generation**: Derives cryptographic hash from encryption key
- **XOR Encryption/Decryption**: Symmetric encryption using keystream derived from SHA-256
- **Key Validation**: Enforces minimum 16-character key length (Requirement 17.8)
- **Length Preservation**: Guarantees encrypted length equals original length (Requirement 24.5)

## Usage

### Basic Usage

```python
from src.encryption_service import EncryptionService

# Initialize service
service = EncryptionService()

# Encrypt text
key = "my_secure_key_16chars"
message = "Secret message"
encrypted = service.encrypt_text(message, key)

# Decrypt text
decrypted = service.decrypt_text(encrypted, key)
assert decrypted == message
```

### Working with Bytes

```python
# Encrypt bytes
data = b"Binary data"
encrypted = service.encrypt(data, key)

# Decrypt bytes
decrypted = service.decrypt(encrypted, key)
assert decrypted == data
```

### Key Validation

```python
# Valid key (>= 16 characters)
service.validate_key("this_is_valid_16c")  # OK

# Invalid key (< 16 characters)
try:
    service.validate_key("short")
except ValueError as e:
    print(e)  # "Encryption key must be at least 16 characters"
```

## API Reference

### `EncryptionService`

#### Methods

##### `validate_key(key: str) -> None`
Validates that the encryption key meets minimum security requirements.

- **Parameters**: 
  - `key`: Encryption key string
- **Raises**: 
  - `ValueError`: If key is less than 16 characters
  - `TypeError`: If key is not a string
- **Requirements**: 17.8

##### `generate_hash(key: str) -> bytes`
Generates SHA-256 hash from encryption key.

- **Parameters**: 
  - `key`: Encryption key (minimum 16 characters)
- **Returns**: 32-byte SHA-256 hash digest
- **Raises**: `ValueError` if key validation fails

##### `encrypt(message: Union[str, bytes], key: str) -> bytes`
Encrypts message using XOR with SHA-256 derived keystream.

- **Parameters**:
  - `message`: Message to encrypt (string or bytes)
  - `key`: Encryption key (minimum 16 characters)
- **Returns**: Encrypted bytes (same length as input)
- **Raises**: `ValueError` if key validation fails
- **Requirements**: 17.8, 24.5

##### `decrypt(encrypted_data: bytes, key: str) -> bytes`
Decrypts data using XOR with SHA-256 derived keystream.

- **Parameters**:
  - `encrypted_data`: Encrypted bytes to decrypt
  - `key`: Encryption key (minimum 16 characters)
- **Returns**: Decrypted bytes (same length as input)
- **Raises**: `ValueError` if key validation fails
- **Requirements**: 17.8, 24.5

##### `encrypt_text(text: str, key: str) -> bytes`
Convenience method to encrypt text and return bytes.

- **Parameters**:
  - `text`: Text message to encrypt
  - `key`: Encryption key (minimum 16 characters)
- **Returns**: Encrypted bytes

##### `decrypt_text(encrypted_data: bytes, key: str) -> str`
Convenience method to decrypt bytes and return text.

- **Parameters**:
  - `encrypted_data`: Encrypted bytes to decrypt
  - `key`: Encryption key (minimum 16 characters)
- **Returns**: Decrypted text string
- **Raises**: `UnicodeDecodeError` if decrypted data is not valid UTF-8

## Requirements Satisfied

### Requirement 17.8: Key Validation
✓ The system validates encryption keys are at least 16 characters
- Implemented in `validate_key()` method
- Enforced in all encryption/decryption operations
- Clear error messages for invalid keys

### Requirement 24.5: Encryption Length Preservation
✓ For all valid messages, the encrypted message length equals the original message length
- XOR encryption preserves length by design
- Verified with assertions in `encrypt()` and `decrypt()` methods
- Tested with various message lengths

## Compatibility

The `EncryptionService` is fully compatible with the existing `backend.steganography.encryption` module:
- Produces identical ciphertext for same inputs
- Can decrypt data encrypted by old module
- Old module can decrypt data encrypted by new service

## Testing

Run tests with:
```bash
# Simple test runner (no pytest required)
python tests/test_encryption_simple.py

# Compatibility tests
python tests/test_compatibility.py

# Full pytest suite (if pytest is configured)
pytest tests/test_encryption_service.py -v
```

## Security Notes

1. **Key Length**: Minimum 16 characters enforced for security
2. **SHA-256**: Cryptographically secure hash function
3. **XOR Properties**: 
   - Symmetric (encrypt = decrypt operation)
   - Length preserving
   - Fast and efficient
4. **Keystream Generation**: Hash is repeated to match message length

## Implementation Details

### Keystream Generation
1. Generate SHA-256 hash from key (32 bytes)
2. Repeat hash to match or exceed message length
3. Truncate to exact message length

### Encryption Process
1. Validate key (>= 16 characters)
2. Convert message to bytes if needed
3. Generate keystream of same length
4. XOR message bytes with keystream bytes
5. Return encrypted bytes

### Decryption Process
1. Validate key (>= 16 characters)
2. Generate keystream of same length as encrypted data
3. XOR encrypted bytes with keystream bytes
4. Return decrypted bytes

## Future Enhancements

Potential improvements for future versions:
- Support for different hash algorithms (SHA-512, etc.)
- Key derivation functions (PBKDF2, Argon2)
- Salt support for additional security
- Configurable minimum key length
