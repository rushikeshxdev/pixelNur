"""
JPEG-Resistant Embedding Strategy for PixelNur Phase 2

This module implements a JPEG-resistant embedding strategy that aligns with
8×8 DCT block boundaries and prioritizes low-frequency DCT components that
survive JPEG quantization.

Key strategies:
1. Align embedding with 8×8 DCT block boundaries to minimize quantization errors
2. Prioritize low-frequency DCT components that survive quantization
3. Avoid high-frequency wavelet sub-bands (HH1, HH2) that are heavily quantized
4. Use standard JPEG quantization tables to predict coefficient survival

Requirements:
- 2.1: System SHALL survive JPEG compression at QF 95 with 100% recovery
- 2.2: System SHALL survive JPEG compression at QF 85 with ≥98% recovery
- 2.3: System SHALL survive JPEG compression at QF 75 with ≥95% recovery

Author: PixelNur Team
Date: 2024
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class JPEGResistantConfig:
    """Configuration for JPEG-resistant embedding."""
    enable_dct_alignment: bool = True
    enable_quantization_prediction: bool = True
    avoid_high_frequency_subbands: bool = True
    min_quality_factor: int = 75  # Minimum QF to optimize for
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_quality_factor < 1 or self.min_quality_factor > 100:
            raise ValueError(
                f"min_quality_factor must be in [1, 100], got {self.min_quality_factor}"
            )


class JPEGResistantEmbedding:
    """
    Implements JPEG-resistant embedding strategy.
    
    This class provides methods to:
    1. Align embedding locations with 8×8 DCT block boundaries
    2. Calculate DCT-based coefficient survival probabilities
    3. Filter wavelet sub-bands to avoid high-frequency components
    4. Generate JPEG-resistant embedding masks
    
    Attributes:
        config: JPEGResistantConfig with strategy parameters
        luminance_qtable: Standard JPEG luminance quantization table
        chrominance_qtable: Standard JPEG chrominance quantization table
    """
    
    # Standard JPEG luminance quantization table (quality factor 50)
    # Used as baseline for predicting coefficient survival
    LUMINANCE_QTABLE_50 = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    
    # Standard JPEG chrominance quantization table (quality factor 50)
    CHROMINANCE_QTABLE_50 = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)
    
    def __init__(self, config: Optional[JPEGResistantConfig] = None):
        """
        Initialize JPEG-resistant embedding strategy.
        
        Args:
            config: Optional JPEGResistantConfig. If None, uses default configuration.
        """
        self.config = config if config is not None else JPEGResistantConfig()
        
        # Scale quantization tables based on target quality factor
        self.luminance_qtable = self._scale_qtable(
            self.LUMINANCE_QTABLE_50,
            self.config.min_quality_factor
        )
        self.chrominance_qtable = self._scale_qtable(
            self.CHROMINANCE_QTABLE_50,
            self.config.min_quality_factor
        )
    
    @staticmethod
    def _scale_qtable(qtable_50: np.ndarray, quality_factor: int) -> np.ndarray:
        """
        Scale quantization table for a specific quality factor.
        
        Uses the standard JPEG quality scaling formula:
        - QF >= 50: scale = (100 - QF) / 50
        - QF < 50: scale = 50 / QF
        
        Args:
            qtable_50: Base quantization table at QF=50
            quality_factor: Target quality factor (1-100)
        
        Returns:
            Scaled quantization table
        
        Requirements: 2.1, 2.2, 2.3 (predict coefficient survival at different QFs)
        """
        if quality_factor < 1:
            quality_factor = 1
        elif quality_factor > 100:
            quality_factor = 100
        
        # Calculate scaling factor
        if quality_factor < 50:
            scale = 50.0 / quality_factor
        else:
            scale = (100.0 - quality_factor) / 50.0
        
        # Scale the quantization table
        scaled_qtable = qtable_50 * scale
        
        # Ensure minimum value of 1
        scaled_qtable = np.maximum(scaled_qtable, 1.0)
        
        return scaled_qtable
    
    def calculate_dct_survival_probability(
        self,
        dct_position: Tuple[int, int],
        use_luminance: bool = True
    ) -> float:
        """
        Calculate the probability that a DCT coefficient survives JPEG compression.
        
        Lower quantization values indicate higher survival probability.
        The survival probability is inversely proportional to the quantization value.
        
        Args:
            dct_position: (row, col) position within 8×8 DCT block (0-7, 0-7)
            use_luminance: If True, use luminance qtable; otherwise chrominance
        
        Returns:
            Survival probability in [0, 1], where 1 = highest survival
        
        Requirements: 2.1, 2.2, 2.3 (use quantization tables to predict survival)
        """
        row, col = dct_position
        
        if row < 0 or row >= 8 or col < 0 or col >= 8:
            raise ValueError(
                f"DCT position must be in [0, 7], got ({row}, {col})"
            )
        
        # Select appropriate quantization table
        qtable = self.luminance_qtable if use_luminance else self.chrominance_qtable
        
        # Get quantization value at this position
        q_value = qtable[row, col]
        
        # Calculate survival probability (inverse of quantization value)
        # Normalize by maximum quantization value for [0, 1] range
        max_q = np.max(qtable)
        survival_prob = 1.0 - (q_value / max_q)
        
        return float(survival_prob)
    
    def get_low_frequency_dct_positions(
        self,
        max_frequency: int = 4
    ) -> List[Tuple[int, int]]:
        """
        Get DCT positions corresponding to low-frequency components.
        
        Low-frequency components are in the top-left corner of the 8×8 DCT block.
        These components have the highest survival probability under JPEG compression.
        
        Args:
            max_frequency: Maximum frequency index (0-7). Default 4 includes
                          positions where row + col <= max_frequency
        
        Returns:
            List of (row, col) positions for low-frequency components
        
        Requirements: 2.1, 2.2, 2.3 (prioritize low-frequency DCT components)
        """
        if max_frequency < 0 or max_frequency > 7:
            raise ValueError(
                f"max_frequency must be in [0, 7], got {max_frequency}"
            )
        
        positions = []
        for row in range(8):
            for col in range(8):
                # Low-frequency components are in top-left corner
                # Use Manhattan distance as frequency metric
                if row + col <= max_frequency:
                    positions.append((row, col))
        
        return positions
    
    def align_to_dct_blocks(
        self,
        coefficients: np.ndarray,
        block_size: int = 8
    ) -> np.ndarray:
        """
        Create a mask that aligns with DCT block boundaries.
        
        This ensures that embedding locations align with 8×8 DCT blocks used
        in JPEG compression, minimizing quantization errors at block boundaries.
        
        Args:
            coefficients: 2D array of wavelet coefficients
            block_size: DCT block size (default: 8 for JPEG)
        
        Returns:
            Binary mask with 1s at DCT-aligned positions, 0s elsewhere
        
        Requirements: 2.1, 2.2, 2.3 (align with 8×8 DCT block boundaries)
        """
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        
        height, width = coefficients.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Mark positions that align with DCT block boundaries
        # These are positions where both row and col are multiples of block_size
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                # Mark a region within each block (not just the corner)
                # Use the low-frequency region of each block
                end_row = min(row + block_size // 2, height)
                end_col = min(col + block_size // 2, width)
                mask[row:end_row, col:end_col] = 1
        
        return mask
    
    def filter_high_frequency_subbands(
        self,
        coefficients_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Filter out high-frequency wavelet sub-bands that are heavily quantized.
        
        Removes HH1 and HH2 (diagonal high-frequency) sub-bands as they are
        most affected by JPEG quantization. Keeps LH and HL sub-bands which
        contain important edge information that survives better.
        
        Args:
            coefficients_dict: Dictionary of sub-band name to coefficient array
        
        Returns:
            Filtered dictionary with HH1 and HH2 removed
        
        Requirements: 2.1, 2.2, 2.3 (avoid high-frequency wavelet sub-bands)
        """
        if not self.config.avoid_high_frequency_subbands:
            return coefficients_dict
        
        # Create filtered dictionary excluding HH1 and HH2
        filtered_dict = {}
        for subband_name, coeffs in coefficients_dict.items():
            # Keep all sub-bands except HH1 and HH2
            if subband_name not in ['HH1', 'HH2']:
                filtered_dict[subband_name] = coeffs
        
        return filtered_dict
    
    def generate_jpeg_resistant_mask(
        self,
        coefficients_dict: Dict[str, np.ndarray],
        base_mask: Optional[np.ndarray] = None,
        survival_threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Generate JPEG-resistant embedding masks for each sub-band.
        
        This method combines multiple strategies:
        1. DCT block alignment
        2. Low-frequency component prioritization
        3. High-frequency sub-band filtering
        4. Survival probability thresholding
        
        Args:
            coefficients_dict: Dictionary of sub-band name to coefficient array
            base_mask: Optional base mask (e.g., from CNN). If provided, the
                      JPEG-resistant mask is combined with it using AND operation
            survival_threshold: Minimum survival probability (0-1) for embedding
        
        Returns:
            Dictionary of sub-band name to binary mask
        
        Requirements: 2.1, 2.2, 2.3 (comprehensive JPEG-resistant strategy)
        """
        if survival_threshold < 0 or survival_threshold > 1:
            raise ValueError(
                f"survival_threshold must be in [0, 1], got {survival_threshold}"
            )
        
        # Filter high-frequency sub-bands
        filtered_coeffs = self.filter_high_frequency_subbands(coefficients_dict)
        
        # Generate masks for each sub-band
        masks = {}
        
        for subband_name, coeffs in filtered_coeffs.items():
            height, width = coeffs.shape
            
            # Start with all ones
            mask = np.ones((height, width), dtype=np.uint8)
            
            # Apply DCT block alignment if enabled
            if self.config.enable_dct_alignment:
                dct_mask = self.align_to_dct_blocks(coeffs)
                mask = mask & dct_mask
            
            # Apply quantization-based survival prediction if enabled
            if self.config.enable_quantization_prediction:
                # Create survival probability mask
                survival_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Get low-frequency DCT positions
                low_freq_positions = self.get_low_frequency_dct_positions(max_frequency=4)
                
                # For each 8×8 block, mark low-frequency positions
                for block_row in range(0, height, 8):
                    for block_col in range(0, width, 8):
                        for dct_row, dct_col in low_freq_positions:
                            row = block_row + dct_row
                            col = block_col + dct_col
                            
                            if row < height and col < width:
                                # Calculate survival probability
                                survival_prob = self.calculate_dct_survival_probability(
                                    (dct_row, dct_col),
                                    use_luminance=True
                                )
                                
                                # Mark if above threshold
                                if survival_prob >= survival_threshold:
                                    survival_mask[row, col] = 1
                
                mask = mask & survival_mask
            
            # Combine with base mask if provided
            if base_mask is not None:
                # Resize base mask to match coefficient dimensions
                if base_mask.shape != coeffs.shape:
                    from scipy.ndimage import zoom
                    scale_h = coeffs.shape[0] / base_mask.shape[0]
                    scale_w = coeffs.shape[1] / base_mask.shape[1]
                    resized_base_mask = zoom(base_mask, (scale_h, scale_w), order=0)
                    resized_base_mask = (resized_base_mask > 0).astype(np.uint8)
                else:
                    resized_base_mask = (base_mask > 0).astype(np.uint8)
                
                mask = mask & resized_base_mask
            
            masks[subband_name] = mask
        
        return masks
    
    def get_jpeg_resistant_subband_weights(self) -> Dict[str, float]:
        """
        Get perceptual weights for sub-bands optimized for JPEG resistance.
        
        These weights prioritize sub-bands that survive JPEG compression better:
        - LH2, HL2: Highest priority (horizontal/vertical edges at level 2)
        - LH1, HL1: Medium priority (horizontal/vertical edges at level 1)
        - HH2: Low priority (diagonal details, more affected by quantization)
        - HH1: Excluded (most affected by JPEG quantization)
        
        Returns:
            Dictionary of sub-band name to weight
        
        Requirements: 2.1, 2.2, 2.3 (prioritize JPEG-resistant sub-bands)
        """
        weights = {
            'LH2': 1.0,   # Horizontal edges - highest priority
            'HL2': 1.0,   # Vertical edges - highest priority
            'LH1': 0.8,   # Horizontal edges level 1 - medium priority
            'HL1': 0.8,   # Vertical edges level 1 - medium priority
        }
        
        # Include HH2 with lower weight if not avoiding high-frequency
        if not self.config.avoid_high_frequency_subbands:
            weights['HH2'] = 0.5
            weights['HH1'] = 0.3
        
        return weights
    
    def estimate_jpeg_capacity(
        self,
        coefficients_dict: Dict[str, np.ndarray],
        base_mask: Optional[np.ndarray] = None,
        robustness_level: str = "none"
    ) -> Dict[str, int]:
        """
        Estimate embedding capacity with JPEG-resistant strategy.
        
        Args:
            coefficients_dict: Dictionary of sub-band name to coefficient array
            base_mask: Optional base mask (e.g., from CNN)
            robustness_level: Robustness level ('none', 'low', 'medium', 'high')
        
        Returns:
            Dictionary with capacity information:
                - 'total_bits': Total available bits
                - 'total_bytes': Total available bytes (after header)
                - 'message_bytes': Maximum message bytes (after ECC overhead)
                - 'subbands_used': List of sub-band names used
        
        Requirements: 2.1, 2.2, 2.3 (capacity estimation with JPEG resistance)
        """
        # Generate JPEG-resistant masks
        masks = self.generate_jpeg_resistant_mask(coefficients_dict, base_mask)
        
        # Count total available bits
        total_bits = 0
        subbands_used = []
        
        for subband_name, mask in masks.items():
            available_bits = np.sum(mask > 0)
            if available_bits > 0:
                total_bits += available_bits
                subbands_used.append(subband_name)
        
        # Subtract header overhead (56 bits)
        header_overhead_bits = 56
        available_bits = total_bits - header_overhead_bits
        
        if available_bits <= 0:
            return {
                'total_bits': 0,
                'total_bytes': 0,
                'message_bytes': 0,
                'subbands_used': []
            }
        
        # Convert to bytes
        available_bytes = available_bits // 8
        
        # Account for ECC overhead
        if robustness_level != 'none':
            from src.robustness_layer import RobustnessLayer
            robustness_layer = RobustnessLayer(robustness_level)
            nsym = robustness_layer._nsym
            max_message_bytes = available_bytes - nsym
            
            if max_message_bytes < 0:
                max_message_bytes = 0
        else:
            max_message_bytes = available_bytes
        
        return {
            'total_bits': total_bits,
            'total_bytes': available_bytes,
            'message_bytes': max_message_bytes,
            'subbands_used': subbands_used
        }


# Convenience functions

def create_jpeg_resistant_embedding(
    min_quality_factor: int = 75,
    enable_dct_alignment: bool = True,
    enable_quantization_prediction: bool = True,
    avoid_high_frequency_subbands: bool = True
) -> JPEGResistantEmbedding:
    """
    Create a JPEG-resistant embedding strategy with custom configuration.
    
    Args:
        min_quality_factor: Minimum JPEG quality factor to optimize for (1-100)
        enable_dct_alignment: Enable DCT block alignment
        enable_quantization_prediction: Enable quantization-based survival prediction
        avoid_high_frequency_subbands: Avoid HH1 and HH2 sub-bands
    
    Returns:
        Configured JPEGResistantEmbedding instance
    
    Requirements: 2.1, 2.2, 2.3 (configurable JPEG-resistant strategy)
    """
    config = JPEGResistantConfig(
        enable_dct_alignment=enable_dct_alignment,
        enable_quantization_prediction=enable_quantization_prediction,
        avoid_high_frequency_subbands=avoid_high_frequency_subbands,
        min_quality_factor=min_quality_factor
    )
    
    return JPEGResistantEmbedding(config)
