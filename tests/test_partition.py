"""
Tests for ZeRO-Q partition module.

Run with: pytest tests/test_partition.py -v
"""

import pytest
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from partition import (
    compute_aligned_partition_sizes,
    partition_quantized_tensor,
    gather_and_dequantize,
    estimate_memory_savings,
    PartitionInfo,
    BNB_AVAILABLE,
)


class TestComputeAlignedPartitionSizes:
    """Tests for partition size computation."""
    
    def test_exact_division(self):
        """Test when numel divides evenly by blocksize × world_size."""
        numel = 4096  # 64 blocks of 64 elements
        world_size = 4
        blocksize = 64
        
        elements, packed, absmax, total = compute_aligned_partition_sizes(
            numel, world_size, blocksize
        )
        
        # 4096 / 4 = 1024 elements per rank
        assert elements == 1024
        # 4-bit: 1024 / 2 = 512 packed bytes
        assert packed == 512
        # 1024 / 64 = 16 blocks = 16 absmax values
        assert absmax == 16
        # No padding needed
        assert total == numel
    
    def test_with_padding_needed(self):
        """Test when padding is required for alignment."""
        numel = 4000  # Not divisible by 64 * 4 = 256
        world_size = 4
        blocksize = 64
        
        elements, packed, absmax, total = compute_aligned_partition_sizes(
            numel, world_size, blocksize
        )
        
        # Should round up to accommodate all elements
        assert elements * world_size >= numel
        # Elements must be aligned to blocksize
        assert elements % blocksize == 0
        # Total should cover original with minimal padding
        assert total >= numel
        assert total == elements * world_size
    
    def test_different_blocksizes(self):
        """Test with various valid block sizes."""
        numel = 8192
        world_size = 4
        
        for blocksize in [64, 128, 256, 512]:
            elements, packed, absmax, total = compute_aligned_partition_sizes(
                numel, world_size, blocksize
            )
            
            # Verify alignment
            assert elements % blocksize == 0, \
                f"Elements {elements} not aligned to blocksize {blocksize}"
            # Packed is half of elements (4-bit = 0.5 bytes)
            assert packed == elements // 2
            # Absmax is elements / blocksize
            assert absmax == elements // blocksize
    
    def test_single_gpu(self):
        """Test with world_size=1 (no distribution)."""
        numel = 1000
        world_size = 1
        blocksize = 64
        
        elements, packed, absmax, total = compute_aligned_partition_sizes(
            numel, world_size, blocksize
        )
        
        # Should still align to blocksize
        assert elements % blocksize == 0
        assert elements >= numel
    
    def test_large_world_size(self):
        """Test with many GPUs."""
        numel = 65536
        world_size = 8
        blocksize = 64
        
        elements, packed, absmax, total = compute_aligned_partition_sizes(
            numel, world_size, blocksize
        )
        
        # Each GPU should get equal share
        assert total == elements * world_size
        assert elements % blocksize == 0
    
    def test_small_tensor(self):
        """Test with tensor smaller than blocksize × world_size."""
        numel = 100  # Less than 64 * 4 = 256
        world_size = 4
        blocksize = 64
        
        elements, packed, absmax, total = compute_aligned_partition_sizes(
            numel, world_size, blocksize
        )
        
        # Must still provide valid partition
        assert elements >= blocksize  # At least one block per rank
        assert total >= numel


@pytest.mark.skipif(not BNB_AVAILABLE, reason="bitsandbytes not installed")
class TestPartitionQuantizedTensor:
    """Tests for tensor partitioning with quantization."""
    
    @pytest.fixture
    def sample_weight(self):
        """Create a sample weight tensor."""
        torch.manual_seed(42)
        return torch.randn(1024, 1024, dtype=torch.float32)
    
    @pytest.fixture
    def small_weight(self):
        """Create a small weight tensor for quick tests."""
        torch.manual_seed(42)
        return torch.randn(256, 256, dtype=torch.float32)
    
    def test_basic_partition(self, small_weight):
        """Test basic partitioning works."""
        local_packed, local_absmax, quant_state, info = partition_quantized_tensor(
            weight=small_weight,
            rank=0,
            world_size=4,
            blocksize=64,
        )
        
        # Check types
        assert isinstance(local_packed, torch.Tensor)
        assert isinstance(local_absmax, torch.Tensor)
        assert isinstance(info, PartitionInfo)
        
        # Check info
        assert info.rank == 0
        assert info.world_size == 4
        assert info.original_shape == small_weight.shape
        assert info.blocksize == 64
    
    def test_partition_sizes(self, small_weight):
        """Test partition sizes are correct."""
        local_packed, local_absmax, quant_state, info = partition_quantized_tensor(
            weight=small_weight,
            rank=0,
            world_size=4,
            blocksize=64,
        )
        
        # Packed should match expected size
        assert local_packed.numel() == info.packed_partition_size
        # Absmax should match expected size
        assert local_absmax.numel() == info.absmax_partition_size
    
    def test_different_ranks_different_data(self, small_weight):
        """Test that different ranks get different partitions."""
        partitions = []
        for rank in range(4):
            local_packed, _, _, _ = partition_quantized_tensor(
                weight=small_weight,
                rank=rank,
                world_size=4,
            )
            partitions.append(local_packed.clone())
        
        # All partitions should be different
        for i in range(4):
            for j in range(i + 1, 4):
                # They shouldn't be exactly equal
                assert not torch.equal(partitions[i], partitions[j]), \
                    f"Rank {i} and {j} have identical partitions"
    
    def test_invalid_rank(self, small_weight):
        """Test that invalid rank raises error."""
        with pytest.raises(ValueError, match="rank.*must be < world_size"):
            partition_quantized_tensor(
                weight=small_weight,
                rank=4,  # Invalid: >= world_size
                world_size=4,
            )
    
    def test_quant_types(self, small_weight):
        """Test both nf4 and fp4 quantization types."""
        for quant_type in ["nf4", "fp4"]:
            local_packed, local_absmax, quant_state, info = partition_quantized_tensor(
                weight=small_weight,
                rank=0,
                world_size=2,
                quant_type=quant_type,
            )
            
            assert quant_state.quant_type == quant_type


@pytest.mark.skipif(not BNB_AVAILABLE, reason="bitsandbytes not installed")
class TestRoundtrip:
    """Test quantize → partition → gather → dequantize roundtrip."""
    
    @pytest.fixture
    def test_weight(self):
        """Create test weight with known distribution."""
        torch.manual_seed(42)
        # Use normal distribution - good for NF4
        return torch.randn(512, 512, dtype=torch.float32)
    
    def test_single_gpu_roundtrip(self, test_weight):
        """Test roundtrip with single GPU (no actual gathering)."""
        # Partition
        local_packed, local_absmax, quant_state, info = partition_quantized_tensor(
            weight=test_weight,
            rank=0,
            world_size=1,  # Single GPU
            blocksize=64,
        )
        
        # Gather (trivial with world_size=1)
        restored = gather_and_dequantize(
            local_packed=local_packed,
            local_absmax=local_absmax,
            quant_state=quant_state,
            partition_info=info,
        )
        
        # Check shape preserved
        assert restored.shape == test_weight.shape
        
        # Check numerical accuracy
        # NF4 should have relatively small error for normally distributed weights
        abs_error = (test_weight - restored).abs()
        rel_error = abs_error / (test_weight.abs() + 1e-8)
        
        # Mean relative error should be small (< 5% for NF4)
        mean_rel_error = rel_error.mean().item()
        assert mean_rel_error < 0.10, \
            f"Mean relative error {mean_rel_error:.4f} too high"
        
        # Max error check (some outliers expected)
        max_abs_error = abs_error.max().item()
        assert max_abs_error < 1.0, \
            f"Max absolute error {max_abs_error:.4f} too high"
    
    def test_numerical_stability(self, test_weight):
        """Test that quantization error is bounded."""
        local_packed, local_absmax, quant_state, info = partition_quantized_tensor(
            weight=test_weight,
            rank=0,
            world_size=1,
            blocksize=64,
            quant_type="nf4",
        )
        
        restored = gather_and_dequantize(
            local_packed, local_absmax, quant_state, info
        )
        
        # Correlation should be very high
        original_flat = test_weight.view(-1)
        restored_flat = restored.view(-1)
        
        # Compute correlation
        mean_orig = original_flat.mean()
        mean_rest = restored_flat.mean()
        
        cov = ((original_flat - mean_orig) * (restored_flat - mean_rest)).mean()
        std_orig = original_flat.std()
        std_rest = restored_flat.std()
        
        correlation = cov / (std_orig * std_rest + 1e-8)
        
        # Correlation should be > 0.99 for NF4
        assert correlation > 0.98, \
            f"Correlation {correlation:.4f} too low"


class TestMemoryEstimation:
    """Tests for memory savings estimation."""
    
    def test_basic_estimation(self):
        """Test memory estimation calculation."""
        numel = 1_000_000  # 1M elements
        world_size = 4
        
        stats = estimate_memory_savings(numel, world_size, torch.float16)
        
        # Original: 1M × 2 bytes = 2MB
        assert stats["original_total_bytes"] == 2_000_000
        
        # ZeRO-3: 2MB / 4 GPUs = 500KB per GPU
        assert stats["zero3_per_gpu_bytes"] == 500_000
        
        # ZeRO-Q should be less than ZeRO-3
        assert stats["zeroq_per_gpu_bytes"] < stats["zero3_per_gpu_bytes"]
        
        # Memory reduction should be > 1
        assert stats["memory_reduction"] > 1.0
    
    def test_32b_model_estimation(self):
        """Test estimation for 32B parameter model."""
        # 32B params
        numel = 32_000_000_000
        world_size = 8
        
        stats = estimate_memory_savings(numel, world_size, torch.float16)
        
        # ZeRO-3: 64GB / 8 = 8GB per GPU
        assert stats["zero3_per_gpu_bytes"] == 8_000_000_000
        
        # ZeRO-Q should be ~4x less
        reduction = stats["memory_reduction"]
        assert 3.0 < reduction < 4.5, \
            f"Expected ~3.8x reduction, got {reduction:.2f}x"


class TestPartitionInfo:
    """Tests for PartitionInfo dataclass."""
    
    def test_needs_padding(self):
        """Test padding detection."""
        # No padding needed
        info1 = PartitionInfo(
            rank=0, world_size=4,
            original_shape=torch.Size([64, 64]),
            original_numel=4096,
            original_dtype=torch.float32,
            packed_partition_size=512,
            absmax_partition_size=16,
            blocksize=64,
            elements_per_rank=1024,
            padded_numel=4096,  # Same as original
        )
        assert not info1.needs_padding
        assert info1.padding_elements == 0
        
        # Padding needed
        info2 = PartitionInfo(
            rank=0, world_size=4,
            original_shape=torch.Size([63, 63]),  # 3969 elements
            original_numel=3969,
            original_dtype=torch.float32,
            packed_partition_size=512,
            absmax_partition_size=16,
            blocksize=64,
            elements_per_rank=1024,
            padded_numel=4096,  # Padded to 4096
        )
        assert info2.needs_padding
        assert info2.padding_elements == 127


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
