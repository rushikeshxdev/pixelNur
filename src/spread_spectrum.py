"""
Spread-Spectrum Embedding Module for PixelNur Phase 2

This module implements spread-spectrum embedding for noise resistance.
Each bit is distributed across N coefficients using a pseudo-random sequence,
making the embedding robust against Gaussian and salt-and-pepper noise.

The spread-spectrum technique averages out noise across multiple coefficients,
allowing correlation-based extraction to recover bits even when individual
coefficients are corrupted.

Requirements:
- 4.5: Implement spread-spectrum embedding for noise resistance
- 4.1: Recover ≥95% of data with Gaussian noise σ=5
- 4.2: Recover ≥90% of data with Gaussian noise σ=10
- 4.3: Recover ≥98% of data with salt-and-pepper noise density 0.01
- 4.4: Recover ≥90% of data with salt-and-pepper noise density 0.05

Author: PixelNur Team
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Optional
import hashlib


class SpreadSpectrumEmbedder:
    """
    Implements spread-spectrum embedding for noise-resistant steganography.
    
    Spread-spectrum embedding distributes each bit across multiple coefficients
    using a pseudo-random sequence. This provides robustness against noise attacks
    by averaging out noise effects across multiple embedding locations.
    
    Algorithm:
    1. Generate pseudo-random sequence S of length N using PRNG seeded with key
    2. For each bit b and coefficients c[i]:
       - If b=1: c'[i] = c[i] + α * S[i]
       - If b=0: c'[i] = c[i] - α * S[i]
    3. During extraction, calculate correlation: corr = Σ(c'[i] * S[i])
       - If corr > 0: extracted_bit = 1
       - If corr < 0: extracted_bit = 0
    
    Attributes:
        spreading_factor (int): Number of coefficients per bit (default: 7)
        key (str): Encryption key used to seed PRNG
        _prng_state (np.random.RandomState): PRNG instance for sequence generation
    """
    
    DEFAULT_SPREADING_FACTOR = 7
    
    def __init__(self, key: str, spreading_factor: int = DEFAULT_SPREADING_FACTOR):
        """
        Initialize spread-spectrum embedder with encryption key.
        
        Args:
            key (str): Encryption key used to seed PRNG (minimum 16 characters)
            spreading_factor (int): Number of coefficients per bit (default: 7)
                                   Higher values provide more robustness but reduce capacity
        
        Raises:
            ValueError: If key is too short or spreading_factor is invalid
        
        Requirements: 4.5 (PRNG seeded with key)
        """
        if len(key) < 16:
            raise ValueError(
                f"Encryption key must be at least 16 characters. "
                f"Provided key length: {len(key)}"
            )
        
        if spreading_factor < 1:
            raise ValueError(
                f"spreading_factor must be at least 1, got {spreading_factor}"
            )
        
        if spreading_factor > 31:
            raise ValueError(
                f"spreading_factor must be at most 31 (practical limit), "
                f"got {spreading_factor}"
            )
        
        self.spreading_factor = spreading_factor
        self.key = key
        
        # Initialize PRNG with seed derived from key
        self._prng_state = self._initialize_prng(key)
    
    def _initialize_prng(self, key: str) -> np.random.RandomState:
        """
        Initialize pseudo-random number generator with seed derived from key.
        
        Uses SHA-256 hash of key to generate a deterministic seed.
        This ensures the same key always produces the same pseudo-random sequence.
        
        Args:
            key (str): Encryption key
        
        Returns:
            np.random.RandomState: Initialized PRNG instance
        
        Requirements: 4.5 (PRNG seeded with encryption key)
        """
        # Generate SHA-256 hash of key
        key_hash = hashlib.sha256(key.encode('utf-8')).digest()
        
        # Convert first 4 bytes to unsigned 32-bit integer for seed
        seed = int.from_bytes(key_hash[:4], byteorder='big')
        
        # Create PRNG with deterministic seed
        prng = np.random.RandomState(seed)
        
        return prng
    
    def generate_sequence(self, length: int) -> np.ndarray:
        """
        Generate pseudo-random sequence of +1 and -1 values.
        
        The sequence is deterministic based on the encryption key, ensuring
        the same sequence is generated during both embedding and extraction.
        
        Args:
            length (int): Length of sequence to generate
        
        Returns:
            np.ndarray: Array of +1 and -1 values with shape (length,)
        
        Requirements: 4.5 (pseudo-random sequence generation)
        """
        if length <= 0:
            raise ValueError(f"Sequence length must be positive, got {length}")
        
        # Generate random bits (0 or 1)
        random_bits = self._prng_state.randint(0, 2, size=length)
        
        # Convert to +1 and -1
        sequence = 2 * random_bits - 1
        
        return sequence.astype(np.float64)
    
    def reset_prng(self) -> None:
        """
        Reset PRNG to initial state.
        
        This is useful when you need to regenerate the same sequence
        from the beginning (e.g., during extraction).
        
        Requirements: 4.5 (deterministic sequence generation)
        """
        self._prng_state = self._initialize_prng(self.key)
    
    def embed_bit(
        self,
        bit: int,
        coefficients: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Embed a single bit across multiple coefficients using spread-spectrum.
        
        Algorithm:
        - Generate pseudo-random sequence S of length N
        - Map bit to message: m = 2*bit - 1 (0→-1, 1→+1)
        - For each coefficient c[i]:
          - c'[i] = c[i] + strength * m * S[i]
        
        This ensures that extraction via correlation works correctly:
        - corr = Σ(c'[i] * S[i]) = Σ(c[i] * S[i]) + strength * m * N
        - The original correlation Σ(c[i] * S[i]) averages to ~0 for random sequences
        - The signal strength * m * N dominates, giving correct sign for extraction
        
        Args:
            bit (int): Bit to embed (0 or 1)
            coefficients (np.ndarray): Array of N coefficients to modify
            strength (float): Embedding strength parameter α
        
        Returns:
            np.ndarray: Modified coefficients with embedded bit
        
        Raises:
            ValueError: If bit is not 0 or 1, or coefficients length mismatch
        
        Requirements: 4.5 (bit spreading across N coefficients)
        """
        if bit not in (0, 1):
            raise ValueError(f"Bit must be 0 or 1, got {bit}")
        
        if len(coefficients) != self.spreading_factor:
            raise ValueError(
                f"Expected {self.spreading_factor} coefficients, "
                f"got {len(coefficients)}"
            )
        
        if strength <= 0:
            raise ValueError(f"Strength must be positive, got {strength}")
        
        # Generate pseudo-random sequence
        sequence = self.generate_sequence(self.spreading_factor)
        
        # Map bit to ±1: bit=0 → m=-1, bit=1 → m=+1
        message = 2 * bit - 1
        
        # Embed using spread-spectrum: c' = c + α * m * S
        modified_coeffs = coefficients + strength * message * sequence
        
        return modified_coeffs
    
    def extract_bit(
        self,
        coefficients: np.ndarray
    ) -> int:
        """
        Extract a bit from coefficients using correlation-based detection.

        Algorithm:
        - Generate the same pseudo-random sequence S used during embedding
        - Center coefficients by subtracting mean to remove DC component
        - Calculate correlation: corr = Σ((c'[i] - mean) * S[i])
        - If corr > 0: bit = 1, else: bit = 0

        Centering the coefficients removes the baseline correlation that
        might exist between the original coefficients and the random sequence,
        making blind extraction more robust.

        The correlation approach is robust to noise because:
        - Noise is averaged out across N coefficients
        - Signal (embedded bit) is coherent with sequence S
        - Noise is incoherent with sequence S

        Args:
            coefficients (np.ndarray): Array of N coefficients to extract from

        Returns:
            int: Extracted bit (0 or 1)

        Raises:
            ValueError: If coefficients length doesn't match spreading_factor

        Requirements: 4.5 (correlation-based extraction for spread bits)
        """
        if len(coefficients) != self.spreading_factor:
            raise ValueError(
                f"Expected {self.spreading_factor} coefficients, "
                f"got {len(coefficients)}"
            )

        # Generate the same pseudo-random sequence
        sequence = self.generate_sequence(self.spreading_factor)

        # Center the coefficients by subtracting mean
        # This removes the DC component that might correlate with the sequence
        centered_coeffs = coefficients - np.mean(coefficients)

        # Calculate correlation with centered coefficients
        correlation = np.sum(centered_coeffs * sequence)

        # Extract bit based on correlation sign
        extracted_bit = 1 if correlation > 0 else 0

        return extracted_bit


    
    def embed_bits(
        self,
        bits: List[int],
        coefficients: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Embed multiple bits using spread-spectrum.
        
        Each bit is embedded across spreading_factor consecutive coefficients.
        Total coefficients required: len(bits) * spreading_factor
        
        Args:
            bits (List[int]): List of bits to embed (each 0 or 1)
            coefficients (np.ndarray): Flat array of coefficients
            strength (float): Embedding strength parameter
        
        Returns:
            np.ndarray: Modified coefficients with embedded bits
        
        Raises:
            ValueError: If insufficient coefficients for all bits
        
        Requirements: 4.5 (spread-spectrum embedding)
        """
        required_coeffs = len(bits) * self.spreading_factor
        
        if len(coefficients) < required_coeffs:
            raise ValueError(
                f"Insufficient coefficients. Required: {required_coeffs}, "
                f"Available: {len(coefficients)}"
            )
        
        # Reset PRNG to ensure consistent sequence
        self.reset_prng()
        
        # Make a copy to avoid modifying original
        modified_coeffs = coefficients.copy()
        
        # Embed each bit
        for i, bit in enumerate(bits):
            start_idx = i * self.spreading_factor
            end_idx = start_idx + self.spreading_factor
            
            # Extract coefficient group for this bit
            coeff_group = modified_coeffs[start_idx:end_idx]
            
            # Embed bit in this group
            modified_group = self.embed_bit(bit, coeff_group, strength)
            
            # Update coefficients
            modified_coeffs[start_idx:end_idx] = modified_group
        
        return modified_coeffs
    
    def extract_bits(
        self,
        coefficients: np.ndarray,
        num_bits: int
    ) -> List[int]:
        """
        Extract multiple bits using correlation-based detection.
        
        Each bit is extracted from spreading_factor consecutive coefficients.
        
        Args:
            coefficients (np.ndarray): Flat array of coefficients
            num_bits (int): Number of bits to extract
        
        Returns:
            List[int]: Extracted bits (each 0 or 1)
        
        Raises:
            ValueError: If insufficient coefficients for requested bits
        
        Requirements: 4.5 (correlation-based extraction)
        """
        required_coeffs = num_bits * self.spreading_factor
        
        if len(coefficients) < required_coeffs:
            raise ValueError(
                f"Insufficient coefficients. Required: {required_coeffs}, "
                f"Available: {len(coefficients)}"
            )
        
        # Reset PRNG to ensure consistent sequence
        self.reset_prng()
        
        extracted_bits = []
        
        # Extract each bit
        for i in range(num_bits):
            start_idx = i * self.spreading_factor
            end_idx = start_idx + self.spreading_factor
            
            # Extract coefficient group for this bit
            coeff_group = coefficients[start_idx:end_idx]
            
            # Extract bit from this group
            bit = self.extract_bit(coeff_group)
            extracted_bits.append(bit)
        
        return extracted_bits
    
    def calculate_capacity_overhead(self) -> float:
        """
        Calculate capacity overhead factor for spread-spectrum embedding.
        
        Spread-spectrum reduces capacity by a factor of spreading_factor
        because each bit requires spreading_factor coefficients.
        
        Returns:
            float: Capacity overhead factor (e.g., 7.0 for spreading_factor=7)
        
        Example:
            >>> embedder = SpreadSpectrumEmbedder("my_secret_key_123", spreading_factor=7)
            >>> embedder.calculate_capacity_overhead()
            7.0
        
        Requirements: 4.5 (capacity impact of spread-spectrum)
        """
        return float(self.spreading_factor)
    
    def get_correlation_confidence(
        self,
        coefficients: np.ndarray
    ) -> float:
        """
        Calculate correlation confidence for extracted bit.
        
        Higher absolute correlation indicates higher confidence in extraction.
        This can be used to detect heavily corrupted bits.
        
        Uses centered coefficients (mean-subtracted) to match the extraction algorithm.
        
        Args:
            coefficients (np.ndarray): Array of N coefficients
        
        Returns:
            float: Correlation value (positive for bit=1, negative for bit=0)
                  Magnitude indicates confidence
        
        Requirements: 4.5 (correlation-based extraction)
        """
        if len(coefficients) != self.spreading_factor:
            raise ValueError(
                f"Expected {self.spreading_factor} coefficients, "
                f"got {len(coefficients)}"
            )
        
        # Generate the same pseudo-random sequence
        sequence = self.generate_sequence(self.spreading_factor)
        
        # Center the coefficients (same as in extract_bit)
        centered_coeffs = coefficients - np.mean(coefficients)
        
        # Calculate correlation with centered coefficients
        correlation = np.sum(centered_coeffs * sequence)
        
        return float(correlation)
    
    def __repr__(self) -> str:
        """String representation of the SpreadSpectrumEmbedder."""
        return (
            f"SpreadSpectrumEmbedder(spreading_factor={self.spreading_factor}, "
            f"key_length={len(self.key)})"
        )


# Convenience functions

def create_spread_spectrum_embedder(
    key: str,
    spreading_factor: int = SpreadSpectrumEmbedder.DEFAULT_SPREADING_FACTOR
) -> SpreadSpectrumEmbedder:
    """
    Create a spread-spectrum embedder with specified parameters.
    
    Args:
        key (str): Encryption key (minimum 16 characters)
        spreading_factor (int): Number of coefficients per bit (default: 7)
    
    Returns:
        SpreadSpectrumEmbedder: Configured embedder instance
    
    Example:
        >>> embedder = create_spread_spectrum_embedder("my_secret_key_123")
        >>> embedder.spreading_factor
        7
    """
    return SpreadSpectrumEmbedder(key, spreading_factor)
