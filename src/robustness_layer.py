"""
Reed-Solomon Error Correction Layer for PixelNur Phase 2

This module implements Reed-Solomon error correction codes (ECC) to provide
robustness against JPEG compression, resizing, and noise attacks.

Three robustness levels are supported:
- Low: RS(255, 223) - corrects 16 byte errors (14.2% overhead)
- Medium: RS(255, 191) - corrects 32 byte errors (33.5% overhead)
- High: RS(255, 127) - corrects 64 byte errors (100% overhead) + 3x replication

High robustness mode includes redundant embedding with 3x replication and
majority voting for maximum attack resistance.

Author: PixelNur Team
Date: 2024
"""

from typing import Dict, Tuple, List, Optional
import reedsolo


class RobustnessLayer:
    """
    Implements Reed-Solomon error correction for steganographic data.
    
    The RobustnessLayer provides configurable error correction capabilities
    to protect hidden data against various attacks including JPEG compression,
    image resizing, and noise addition.
    
    High robustness mode includes 3x redundant embedding with majority voting
    for maximum attack resistance.
    
    Attributes:
        robustness_level (str): Current robustness level ('none', 'low', 'medium', 'high')
        _rs_codec (reedsolo.RSCodec): Reed-Solomon codec instance
        _nsym (int): Number of error correction symbols
        _max_correctable (int): Maximum correctable byte errors
        _replication_factor (int): Number of replicas for redundant embedding (3 for high, 1 otherwise)
    """
    
    # Reed-Solomon configuration for each robustness level
    # Format: (n_symbols, max_correctable_errors, overhead_percentage, replication_factor)
    ROBUSTNESS_CONFIGS = {
        'none': (0, 0, 0.0, 1),
        'low': (32, 16, 14.2, 1),      # RS(255, 223): 32 parity bytes
        'medium': (64, 32, 33.5, 1),   # RS(255, 191): 64 parity bytes
        'high': (128, 64, 100.0, 3)    # RS(255, 127): 128 parity bytes + 3x replication
    }
    
    # Sub-band distribution for replicas in high robustness mode
    # Distributes replicas across different LWT sub-bands for resilience
    REPLICA_SUBBAND_DISTRIBUTION = {
        0: ['LH2', 'HL2'],  # Replica 1: horizontal/vertical details at level 2
        1: ['HH2', 'LH1'],  # Replica 2: diagonal details at level 2, horizontal at level 1
        2: ['HL1', 'HH1']   # Replica 3: vertical and diagonal details at level 1
    }
    
    def __init__(self, robustness_level: str = 'none'):
        """
        Initialize the RobustnessLayer with specified robustness level.
        
        Args:
            robustness_level (str): One of 'none', 'low', 'medium', 'high'
                                   Defaults to 'none' (no error correction)
        
        Raises:
            ValueError: If robustness_level is not recognized
        """
        if robustness_level not in self.ROBUSTNESS_CONFIGS:
            raise ValueError(
                f"Invalid robustness level: {robustness_level}. "
                f"Must be one of {list(self.ROBUSTNESS_CONFIGS.keys())}"
            )
        
        self.robustness_level = robustness_level
        self._nsym, self._max_correctable, self._overhead, self._replication_factor = \
            self.ROBUSTNESS_CONFIGS[robustness_level]
        
        # Initialize Reed-Solomon codec if error correction is enabled
        if self._nsym > 0:
            self._rs_codec = reedsolo.RSCodec(nsym=self._nsym)
        else:
            self._rs_codec = None
    
    def encode(self, message: bytes) -> bytes:
        """
        Apply Reed-Solomon error correction encoding to the message.
        
        For high robustness mode, this only applies ECC encoding.
        Use encode_with_replication() for 3x replication.
        
        Args:
            message (bytes): Original message to encode
        
        Returns:
            bytes: Encoded message with error correction symbols appended
                  If robustness_level is 'none', returns original message unchanged
        
        Example:
            >>> layer = RobustnessLayer('low')
            >>> encoded = layer.encode(b'Hello, World!')
            >>> len(encoded) > len(b'Hello, World!')
            True
        """
        if self._rs_codec is None:
            # No error correction - return message as-is
            return message
        
        # Apply Reed-Solomon encoding
        encoded_message = self._rs_codec.encode(message)
        return encoded_message
    
    def encode_with_replication(self, message: bytes) -> List[bytes]:
        """
        Apply Reed-Solomon encoding and create replicas for redundant embedding.
        
        For high robustness mode (replication_factor=3), creates 3 identical
        encoded replicas to be distributed across different LWT sub-bands.
        For other modes, returns a single-item list with the encoded message.
        
        Args:
            message (bytes): Original message to encode
        
        Returns:
            List[bytes]: List of encoded message replicas
                        Length equals replication_factor (1 for low/medium, 3 for high)
        
        Example:
            >>> layer = RobustnessLayer('high')
            >>> replicas = layer.encode_with_replication(b'Secret data')
            >>> len(replicas)
            3
            >>> all(r == replicas[0] for r in replicas)  # All replicas identical
            True
        
        Requirements:
            - 2.8: Use redundant embedding with 3x replication for high robustness
        """
        # Apply ECC encoding
        encoded_message = self.encode(message)
        
        # Create replicas based on replication factor
        replicas = [encoded_message] * self._replication_factor
        
        return replicas
    
    def decode(self, encoded_message: bytes) -> Tuple[bytes, Dict]:
        """
        Apply Reed-Solomon error correction decoding to recover the original message.
        
        For high robustness mode with replicas, use decode_with_majority_voting() instead.
        
        Args:
            encoded_message (bytes): Encoded message with potential errors
        
        Returns:
            Tuple[bytes, Dict]: A tuple containing:
                - decoded_message (bytes): Recovered original message
                - stats (Dict): Error correction statistics with keys:
                    - 'errors_corrected' (int): Number of byte errors corrected
                    - 'errors_detected' (int): Number of uncorrectable errors detected
                    - 'success' (bool): True if decoding succeeded, False otherwise
                    - 'error_message' (str): Error description if decoding failed
        
        Example:
            >>> layer = RobustnessLayer('low')
            >>> encoded = layer.encode(b'Test message')
            >>> decoded, stats = layer.decode(encoded)
            >>> stats['success']
            True
            >>> stats['errors_corrected']
            0
        """
        stats = {
            'errors_corrected': 0,
            'errors_detected': 0,
            'success': True,
            'error_message': ''
        }
        
        if self._rs_codec is None:
            # No error correction - return message as-is
            return encoded_message, stats
        
        try:
            # Attempt Reed-Solomon decoding with error correction
            # decode() returns (decoded_message, decoded_msg, errata_pos)
            result = self._rs_codec.decode(encoded_message)
            
            # Handle different return formats from reedsolo
            if isinstance(result, tuple) and len(result) == 3:
                decoded_message, decoded_msg, errata_pos = result
            elif isinstance(result, tuple) and len(result) == 2:
                decoded_message, errata_pos = result
            else:
                decoded_message = result
                errata_pos = []
            
            # Count corrected errors
            if errata_pos:
                stats['errors_corrected'] = len(errata_pos)
            
        except reedsolo.ReedSolomonError as e:
            # Decoding failed - too many errors to correct
            stats['success'] = False
            stats['error_message'] = str(e)
            stats['errors_detected'] = self._max_correctable + 1  # Exceeded capacity
            
            # Return the message without error correction
            # Strip parity symbols to get corrupted original data
            if len(encoded_message) > self._nsym:
                decoded_message = encoded_message[:-self._nsym]
            else:
                decoded_message = encoded_message
        
        return decoded_message, stats
    
    def decode_with_majority_voting(self, replicas: List[bytes]) -> Tuple[bytes, Dict]:
        """
        Decode multiple replicas and use majority voting to recover the message.
        
        This method is designed for high robustness mode where 3 replicas are
        extracted from different LWT sub-bands. Each replica is decoded independently,
        and majority voting is applied byte-by-byte to recover the most likely
        original message.
        
        Majority voting algorithm:
        1. Decode each replica independently using Reed-Solomon ECC
        2. For each byte position, count votes across all replicas
        3. Select the byte value with the most votes (majority wins)
        4. If all replicas disagree, use the first successfully decoded replica
        
        Args:
            replicas (List[bytes]): List of encoded message replicas extracted
                                   from different sub-bands
        
        Returns:
            Tuple[bytes, Dict]: A tuple containing:
                - decoded_message (bytes): Recovered message using majority voting
                - stats (Dict): Comprehensive statistics with keys:
                    - 'replicas_decoded' (int): Number of successfully decoded replicas
                    - 'replicas_failed' (int): Number of failed replica decodings
                    - 'majority_votes_used' (int): Number of bytes decided by majority
                    - 'unanimous_bytes' (int): Number of bytes where all replicas agreed
                    - 'errors_corrected_per_replica' (List[int]): Errors corrected in each replica
                    - 'success' (bool): True if at least one replica decoded successfully
                    - 'error_message' (str): Error description if all replicas failed
        
        Example:
            >>> layer = RobustnessLayer('high')
            >>> replicas = layer.encode_with_replication(b'Secret')
            >>> decoded, stats = layer.decode_with_majority_voting(replicas)
            >>> stats['success']
            True
            >>> stats['replicas_decoded']
            3
        
        Requirements:
            - 2.8: Implement majority voting during extraction for high robustness
        """
        stats = {
            'replicas_decoded': 0,
            'replicas_failed': 0,
            'majority_votes_used': 0,
            'unanimous_bytes': 0,
            'errors_corrected_per_replica': [],
            'success': False,
            'error_message': ''
        }
        
        # Handle single replica case (low/medium robustness)
        if len(replicas) == 1:
            decoded, decode_stats = self.decode(replicas[0])
            stats['replicas_decoded'] = 1 if decode_stats['success'] else 0
            stats['replicas_failed'] = 0 if decode_stats['success'] else 1
            stats['errors_corrected_per_replica'] = [decode_stats['errors_corrected']]
            stats['success'] = decode_stats['success']
            stats['error_message'] = decode_stats['error_message']
            return decoded, stats
        
        # Decode all replicas
        decoded_replicas = []
        for replica in replicas:
            decoded, decode_stats = self.decode(replica)
            if decode_stats['success']:
                decoded_replicas.append(decoded)
                stats['replicas_decoded'] += 1
            else:
                stats['replicas_failed'] += 1
            stats['errors_corrected_per_replica'].append(decode_stats['errors_corrected'])
        
        # Check if at least one replica decoded successfully
        if not decoded_replicas:
            stats['error_message'] = 'All replicas failed to decode'
            return b'', stats
        
        stats['success'] = True
        
        # If only one replica decoded, return it
        if len(decoded_replicas) == 1:
            return decoded_replicas[0], stats
        
        # Apply majority voting byte-by-byte
        # Find the maximum length among decoded replicas
        max_length = max(len(r) for r in decoded_replicas)
        
        # Pad shorter replicas with zeros for voting
        padded_replicas = []
        for replica in decoded_replicas:
            if len(replica) < max_length:
                padded_replicas.append(replica + b'\x00' * (max_length - len(replica)))
            else:
                padded_replicas.append(replica)
        
        # Perform majority voting
        result_bytes = []
        for byte_idx in range(max_length):
            # Count votes for each byte value at this position
            votes = {}
            for replica in padded_replicas:
                byte_val = replica[byte_idx]
                votes[byte_val] = votes.get(byte_val, 0) + 1
            
            # Find the byte value with the most votes
            majority_byte = max(votes.keys(), key=lambda k: votes[k])
            result_bytes.append(majority_byte)
            
            # Track statistics
            if votes[majority_byte] > 1:
                stats['majority_votes_used'] += 1
            if votes[majority_byte] == len(padded_replicas):
                stats['unanimous_bytes'] += 1
        
        decoded_message = bytes(result_bytes)
        
        return decoded_message, stats
    
    def calculate_overhead(self) -> float:
        """
        Calculate the overhead percentage for the current robustness level.
        
        For high robustness mode, this includes both ECC overhead (100%) and
        replication overhead (200% for 3x replication), totaling 300% overhead.
        
        Returns:
            float: Overhead percentage (e.g., 14.2 for low robustness, 300.0 for high)
                  Returns 0.0 if robustness_level is 'none'
        
        Example:
            >>> layer = RobustnessLayer('medium')
            >>> layer.calculate_overhead()
            33.5
            >>> layer_high = RobustnessLayer('high')
            >>> layer_high.calculate_overhead()
            300.0
        """
        if self.robustness_level == 'high':
            # High robustness: ECC overhead (100%) + replication overhead (200%)
            # 3x replication means 2 extra copies = 200% overhead
            return self._overhead + (self._replication_factor - 1) * 100.0
        return self._overhead
    
    def get_replication_factor(self) -> int:
        """
        Get the replication factor for the current robustness level.
        
        Returns:
            int: Number of replicas (1 for none/low/medium, 3 for high)
        
        Example:
            >>> layer = RobustnessLayer('high')
            >>> layer.get_replication_factor()
            3
            >>> layer_low = RobustnessLayer('low')
            >>> layer_low.get_replication_factor()
            1
        
        Requirements:
            - 2.8: Support 3x replication for high robustness level
        """
        return self._replication_factor
    
    def get_replica_subband_distribution(self) -> Optional[Dict[int, List[str]]]:
        """
        Get the sub-band distribution strategy for replicas.
        
        Returns:
            Optional[Dict[int, List[str]]]: Dictionary mapping replica index to
                                           preferred LWT sub-bands, or None if
                                           replication is not enabled
        
        Example:
            >>> layer = RobustnessLayer('high')
            >>> dist = layer.get_replica_subband_distribution()
            >>> dist[0]
            ['LH2', 'HL2']
            >>> dist[1]
            ['HH2', 'LH1']
            >>> dist[2]
            ['HL1', 'HH1']
        
        Requirements:
            - 2.8: Distribute replicas across different LWT sub-bands
        """
        if self._replication_factor > 1:
            return self.REPLICA_SUBBAND_DISTRIBUTION
        return None
    
    def get_max_correctable_errors(self) -> int:
        """
        Get the maximum number of byte errors that can be corrected.
        
        Returns:
            int: Maximum correctable byte errors for current robustness level
                Returns 0 if robustness_level is 'none'
        
        Example:
            >>> layer = RobustnessLayer('high')
            >>> layer.get_max_correctable_errors()
            64
        """
        return self._max_correctable
    
    def get_encoded_length(self, message_length: int) -> int:
        """
        Calculate the length of the encoded message given the original message length.
        
        This accounts for ECC overhead but NOT replication overhead.
        For total capacity with replication, multiply by replication_factor.
        
        Args:
            message_length (int): Length of the original message in bytes
        
        Returns:
            int: Length of the encoded message including error correction symbols
        
        Example:
            >>> layer = RobustnessLayer('low')
            >>> layer.get_encoded_length(100)
            132
        """
        return message_length + self._nsym
    
    def is_redundant_embedding_enabled(self) -> bool:
        """
        Check if redundant embedding (replication) is enabled.
        
        Returns:
            bool: True if replication_factor > 1 (high robustness mode), False otherwise
        
        Example:
            >>> layer = RobustnessLayer('high')
            >>> layer.is_redundant_embedding_enabled()
            True
            >>> layer_low = RobustnessLayer('low')
            >>> layer_low.is_redundant_embedding_enabled()
            False
        
        Requirements:
            - 2.8: Support enabling/disabling redundant embedding
        """
        return self._replication_factor > 1
    
    def __repr__(self) -> str:
        """String representation of the RobustnessLayer."""
        if self._replication_factor > 1:
            return (
                f"RobustnessLayer(level='{self.robustness_level}', "
                f"max_correctable={self._max_correctable}, "
                f"overhead={self._overhead}%, "
                f"replication={self._replication_factor}x)"
            )
        return (
            f"RobustnessLayer(level='{self.robustness_level}', "
            f"max_correctable={self._max_correctable}, "
            f"overhead={self._overhead}%)"
        )
