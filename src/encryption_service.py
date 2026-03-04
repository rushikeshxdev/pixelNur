"""
Encryption Service for PixelNur Phase 2
Implements Slice Encryption using SHA-256 + XOR

Requirements:
- 17.8: Validate encryption keys are at least 16 characters
- 24.5: Encrypted message length SHALL equal original message length (encryption invariant)
"""

import hashlib
from typing import Union


class EncryptionService:
    """
    Provides encryption and decryption services using Slice Encryption (SHA-256 + XOR).
    
    This service ensures:
    - Key validation (minimum 16 characters)
    - Length preservation (encrypted length = original length)
    - Secure SHA-256 hash generation
    """
    
    MIN_KEY_LENGTH = 16
    
    def __init__(self):
        """Initialize the EncryptionService."""
        pass
    
    def validate_key(self, key: str) -> None:
        """
        Validate encryption key meets minimum security requirements.
        
        Args:
            key: The encryption key to validate
            
        Raises:
            ValueError: If key is less than 16 characters
            
        Requirements: 17.8
        """
        if not isinstance(key, str):
            raise TypeError("Encryption key must be a string")
        
        if len(key) < self.MIN_KEY_LENGTH:
            raise ValueError(
                f"Encryption key must be at least {self.MIN_KEY_LENGTH} characters. "
                f"Provided key length: {len(key)}"
            )
    
    def generate_hash(self, key: str) -> bytes:
        """
        Generate SHA-256 hash from encryption key.
        
        Args:
            key: The encryption key
            
        Returns:
            32-byte SHA-256 hash digest
            
        Raises:
            ValueError: If key validation fails
        """
        self.validate_key(key)
        return hashlib.sha256(key.encode('utf-8')).digest()
    
    def _generate_keystream(self, key: str, length: int) -> bytes:
        """
        Generate a keystream of specified length by repeating SHA-256 hash.
        
        Args:
            key: The encryption key
            length: Required keystream length in bytes
            
        Returns:
            Keystream bytes of specified length
        """
        hash_digest = self.generate_hash(key)
        
        # Calculate how many times we need to repeat the hash
        repeats = (length // len(hash_digest)) + 1
        
        # Repeat and truncate to exact length
        keystream = (hash_digest * repeats)[:length]
        
        return keystream
    
    def encrypt(self, message: Union[str, bytes], key: str) -> bytes:
        """
        Encrypt message using XOR with SHA-256 derived keystream.
        
        Args:
            message: The message to encrypt (string or bytes)
            key: The encryption key (minimum 16 characters)
            
        Returns:
            Encrypted bytes (same length as input)
            
        Raises:
            ValueError: If key validation fails
            
        Requirements: 17.8, 24.5
        """
        self.validate_key(key)
        
        # Convert message to bytes if it's a string
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        else:
            message_bytes = message
        
        # Generate keystream of same length as message
        keystream = self._generate_keystream(key, len(message_bytes))
        
        # XOR encryption
        encrypted = bytes(a ^ b for a, b in zip(message_bytes, keystream))
        
        # Verify length preservation (Requirement 24.5)
        assert len(encrypted) == len(message_bytes), \
            "Encryption length preservation invariant violated"
        
        return encrypted
    
    def decrypt(self, encrypted_data: bytes, key: str) -> bytes:
        """
        Decrypt data using XOR with SHA-256 derived keystream.
        
        XOR is symmetric, so decryption is identical to encryption.
        
        Args:
            encrypted_data: The encrypted bytes to decrypt
            key: The encryption key (minimum 16 characters)
            
        Returns:
            Decrypted bytes (same length as input)
            
        Raises:
            ValueError: If key validation fails
            
        Requirements: 17.8, 24.5
        """
        self.validate_key(key)
        
        # Generate keystream of same length as encrypted data
        keystream = self._generate_keystream(key, len(encrypted_data))
        
        # XOR decryption (same as encryption due to XOR properties)
        decrypted = bytes(a ^ b for a, b in zip(encrypted_data, keystream))
        
        # Verify length preservation (Requirement 24.5)
        assert len(decrypted) == len(encrypted_data), \
            "Decryption length preservation invariant violated"
        
        return decrypted
    
    def encrypt_text(self, text: str, key: str) -> bytes:
        """
        Convenience method to encrypt text and return bytes.
        
        Args:
            text: The text message to encrypt
            key: The encryption key (minimum 16 characters)
            
        Returns:
            Encrypted bytes
        """
        return self.encrypt(text, key)
    
    def decrypt_text(self, encrypted_data: bytes, key: str) -> str:
        """
        Convenience method to decrypt bytes and return text.
        
        Args:
            encrypted_data: The encrypted bytes to decrypt
            key: The encryption key (minimum 16 characters)
            
        Returns:
            Decrypted text string
            
        Raises:
            UnicodeDecodeError: If decrypted data is not valid UTF-8
        """
        decrypted_bytes = self.decrypt(encrypted_data, key)
        return decrypted_bytes.decode('utf-8')
