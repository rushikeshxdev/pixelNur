"""
Scale-Invariant Embedding Module for PixelNur Phase 2

Requirements: 3.5, 3.6, 3.7, 3.8
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import zoom


class ScaleInvariantEmbedder:
    """Scale-invariant embedder for resize-resistant steganography."""
    
    DEFAULT_REDUNDANCY_FACTOR = 2
    DEFAULT_RATIO_THRESHOLD = 1.0
    LOW_FREQ_SUBBANDS = ['LL2', 'LH2', 'HL2', 'LH1', 'HL1']
    
    def __init__(self, redundancy_factor=DEFAULT_REDUNDANCY_FACTOR, ratio_threshold=DEFAULT_RATIO_THRESHOLD):
        if redundancy_factor < 1 or redundancy_factor > 5:
            raise ValueError(f'redundancy_factor must be between 1 and 5, got {redundancy_factor}')
        if ratio_threshold <= 0:
            raise ValueError(f'ratio_threshold must be positive, got {ratio_threshold}')
        self.redundancy_factor = redundancy_factor
        self.ratio_threshold = ratio_threshold
        self.low_freq_subbands = self.LOW_FREQ_SUBBANDS.copy()
    
    def _create_coefficient_pairs(self, coefficients):
        height, width = coefficients.shape
        pairs = []
        for i in range(height):
            for j in range(0, width - 1, 2):
                pairs.append((i, j, i, j + 1))
        for i in range(0, height - 1, 2):
            for j in range(width):
                pairs.append((i, j, i + 1, j))
        return pairs
    
    def embed_bit_in_ratio(self, coeff1, coeff2, bit, strength):
        if bit not in (0, 1):
            raise ValueError(f'Bit must be 0 or 1, got {bit}')
        if abs(coeff2) < 1e-6:
            coeff2 = 1e-6 if coeff2 >= 0 else -1e-6
        if strength <= 0:
            raise ValueError(f'Strength must be positive, got {strength}')
        
        target_ratio = self.ratio_threshold * 1.5 if bit == 1 else self.ratio_threshold / 1.5
        target_coeff1 = target_ratio * coeff2
        delta1 = (target_coeff1 - coeff1) * strength
        
        return coeff1 + delta1, coeff2
    
    def extract_bit_from_ratio(self, coeff1, coeff2):
        if abs(coeff2) < 1e-6:
            return 1 if coeff1 > 0 else 0
        ratio = coeff1 / coeff2
        return 1 if ratio > self.ratio_threshold else 0
    
    def embed_with_redundancy(self, coefficients_dict, bits, strength):
        modified_coeffs = {k: v.copy() for k, v in coefficients_dict.items()}
        total_bits_to_embed = len(bits) * self.redundancy_factor
        
        all_pairs = []
        for subband_name in self.low_freq_subbands:
            if subband_name not in modified_coeffs:
                continue
            coeffs = modified_coeffs[subband_name]
            pairs = self._create_coefficient_pairs(coeffs)
            for pair in pairs:
                all_pairs.append((subband_name, pair))
        
        if len(all_pairs) < total_bits_to_embed:
            raise ValueError(f'Insufficient capacity: need {total_bits_to_embed}, have {len(all_pairs)}')
        
        pair_idx = 0
        for bit in bits:
            for replica in range(self.redundancy_factor):
                subband_name, (row1, col1, row2, col2) = all_pairs[pair_idx]
                pair_idx += 1
                
                coeffs = modified_coeffs[subband_name]
                coeff1, coeff2 = coeffs[row1, col1], coeffs[row2, col2]
                modified_coeff1, modified_coeff2 = self.embed_bit_in_ratio(coeff1, coeff2, bit, strength)
                
                coeffs[row1, col1] = modified_coeff1
                coeffs[row2, col2] = modified_coeff2
        
        return modified_coeffs
    
    def extract_with_majority_voting(self, coefficients_dict, num_bits):
        all_pairs = []
        for subband_name in self.low_freq_subbands:
            if subband_name not in coefficients_dict:
                continue
            coeffs = coefficients_dict[subband_name]
            pairs = self._create_coefficient_pairs(coeffs)
            for pair in pairs:
                all_pairs.append((subband_name, pair))
        
        total_pairs_needed = num_bits * self.redundancy_factor
        if len(all_pairs) < total_pairs_needed:
            raise ValueError(f'Insufficient data: need {total_pairs_needed}, have {len(all_pairs)}')
        
        extracted_bits = []
        pair_idx = 0
        
        for bit_idx in range(num_bits):
            bit_votes = []
            for replica in range(self.redundancy_factor):
                subband_name, (row1, col1, row2, col2) = all_pairs[pair_idx]
                pair_idx += 1
                
                coeffs = coefficients_dict[subband_name]
                bit = self.extract_bit_from_ratio(coeffs[row1, col1], coeffs[row2, col2])
                bit_votes.append(bit)
            
            majority_bit = 1 if sum(bit_votes) > len(bit_votes) / 2 else 0
            extracted_bits.append(majority_bit)
        
        return extracted_bits
    
    def detect_resize(self, stego_image, expected_shape):
        current_shape = stego_image.shape[:2]
        expected_h, expected_w = expected_shape
        current_h, current_w = current_shape
        
        if current_h == expected_h and current_w == expected_w:
            return False, None
        
        return True, (current_h / expected_h, current_w / expected_w)
    
    def apply_scale_compensation(self, stego_image, target_shape, interpolation_method='bilinear'):
        valid_methods = ['bilinear', 'bicubic', 'nearest']
        if interpolation_method not in valid_methods:
            raise ValueError(f'Invalid interpolation method: {interpolation_method}')
        
        current_shape = stego_image.shape[:2]
        target_h, target_w = target_shape
        current_h, current_w = current_shape
        
        zoom_h, zoom_w = target_h / current_h, target_w / current_w
        order_map = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}
        order = order_map[interpolation_method]
        
        if stego_image.ndim == 2:
            return zoom(stego_image, (zoom_h, zoom_w), order=order)
        elif stego_image.ndim == 3:
            return zoom(stego_image, (zoom_h, zoom_w, 1), order=order)
        else:
            raise ValueError(f'Unsupported image dimensions: {stego_image.shape}')
    
    def calculate_capacity_overhead(self):
        return 2.0 * self.redundancy_factor
    
    def get_voting_confidence(self, coefficients_dict, bit_index):
        all_pairs = []
        for subband_name in self.low_freq_subbands:
            if subband_name not in coefficients_dict:
                continue
            coeffs = coefficients_dict[subband_name]
            pairs = self._create_coefficient_pairs(coeffs)
            for pair in pairs:
                all_pairs.append((subband_name, pair))
        
        start_pair_idx = bit_index * self.redundancy_factor
        if start_pair_idx + self.redundancy_factor > len(all_pairs):
            raise ValueError(f'Bit index {bit_index} out of range')
        
        bit_votes = []
        for replica in range(self.redundancy_factor):
            pair_idx = start_pair_idx + replica
            subband_name, (row1, col1, row2, col2) = all_pairs[pair_idx]
            coeffs = coefficients_dict[subband_name]
            bit = self.extract_bit_from_ratio(coeffs[row1, col1], coeffs[row2, col2])
            bit_votes.append(bit)
        
        majority_bit = 1 if sum(bit_votes) > len(bit_votes) / 2 else 0
        agreement_count = sum(1 for vote in bit_votes if vote == majority_bit)
        return agreement_count / len(bit_votes)
    
    def __repr__(self):
        return f'ScaleInvariantEmbedder(redundancy_factor={self.redundancy_factor}, ratio_threshold={self.ratio_threshold})'


def create_scale_invariant_embedder(redundancy_factor=2, ratio_threshold=1.0):
    return ScaleInvariantEmbedder(redundancy_factor, ratio_threshold)
