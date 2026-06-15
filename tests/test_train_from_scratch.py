"""
Tests for ZeRO-Q training-from-scratch mode.

Run with: pytest tests/test_train_from_scratch.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partition import partition_fp32, gather_fp32

try:
    from config import ZeroQTrainConfig, TRAIN_FROM_SCRATCH_CONFIG
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

try:
    from gradient_sync import reduce_scatter_grads
    HAS_GRADIENT_SYNC = True
except ImportError:
    HAS_GRADIENT_SYNC = False

try:
    from coordinator import ZeroQParameter, ZeroQCoordinator, ZeroQModuleWrapper
    HAS_COORDINATOR = True
except ImportError:
    HAS_COORDINATOR = False


class TestPartitionFp32:
    """Tests for partition_fp32() and gather_fp32()."""

    def test_roundtrip_single_rank(self):
        """Shard -> gather with world_size=1 is bit-exact."""
        torch.manual_seed(42)
        original = torch.randn(128, 64, dtype=torch.float32)
        shard, shape = partition_fp32(original, world_size=1, rank=0)
        restored = gather_fp32(shard, shape, world_size=1)
        assert torch.equal(original, restored)

    def test_shard_size_exact_division(self):
        """Shard size matches numel // world_size when evenly divisible."""
        original = torch.randn(256, dtype=torch.float32)
        for ws in [1, 2, 4, 8]:
            shard, _ = partition_fp32(original, world_size=ws, rank=0)
            assert shard.numel() == (256 + ws - 1) // ws

    def test_padding_when_not_divisible(self):
        """Last rank's shard is padded to match chunk_size."""
        original = torch.randn(100, dtype=torch.float32)
        chunk_size = (100 + 3 - 1) // 3
        for rank in range(3):
            shard, _ = partition_fp32(original, world_size=3, rank=rank)
            assert shard.numel() == chunk_size

    def test_dtype_conversion(self):
        """Input in fp16 is cast to fp32."""
        original = torch.randn(64, dtype=torch.float16)
        shard, _ = partition_fp32(original, world_size=1, rank=0)
        assert shard.dtype == torch.float32

    def test_multi_rank_simulation(self):
        """Simulate multiple ranks and verify full reconstruction."""
        torch.manual_seed(7)
        original = torch.randn(100, 50, dtype=torch.float32)
        world_size = 4
        shards = []
        for rank in range(world_size):
            shard, shape = partition_fp32(original, world_size=world_size, rank=rank)
            shards.append(shard)
        full = torch.cat(shards)[: original.numel()]
        reconstructed = full.view(original.shape)
        assert torch.equal(original, reconstructed)

    def test_different_shapes(self):
        """partition_fp32 works for 1-D, 2-D, 3-D tensors."""
        for shape in [(512,), (64, 64), (8, 16, 32)]:
            t = torch.randn(shape, dtype=torch.float32)
            shard, s = partition_fp32(t, world_size=2, rank=0)
            chunk = (t.numel() + 1) // 2
            assert shard.numel() == chunk
            assert s == torch.Size(shape)


@pytest.mark.skipif(not HAS_GRADIENT_SYNC, reason="gradient_sync not available")
class TestReduceScatterGrads:
    """Tests for reduce_scatter_grads()."""

    def test_single_rank_returns_full_grad(self):
        """world_size=1: returns clone of full gradient (flattened)."""
        grad = torch.randn(32, 16, dtype=torch.float32)
        result = reduce_scatter_grads(grad, world_size=1, rank=0)
        expected = grad.contiguous().view(-1)
        assert torch.equal(result, expected)

    def test_output_shape(self):
        """Output shape matches chunk_size for world_size=1."""
        grad = torch.randn(100, dtype=torch.float32)
        result = reduce_scatter_grads(grad, world_size=1, rank=0)
        assert result.shape == (100,)

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""
        grad = torch.randn(64, dtype=torch.float32)
        result = reduce_scatter_grads(grad, world_size=1, rank=0)
        assert result.dtype == torch.float32


@pytest.mark.skipif(not HAS_CONFIG, reason="config not available")
class TestZeroQTrainConfig:
    """Tests for ZeroQTrainConfig."""

    def test_defaults(self):
        config = ZeroQTrainConfig()
        assert config.training_mode is True
        assert config.frozen_only is False
        assert config.partition_trainable is True
        assert config.compress_between_steps is False
        assert config.optimizer_cls == "AdamW"
        assert config.optimizer_kwargs == {"lr": 3e-4}

    def test_preset(self):
        assert TRAIN_FROM_SCRATCH_CONFIG.training_mode is True
        assert TRAIN_FROM_SCRATCH_CONFIG.frozen_only is False
        assert TRAIN_FROM_SCRATCH_CONFIG.compute_dtype == torch.float32

    def test_no_validation_without_compress(self):
        """No quant validation when compress_between_steps=False."""
        config = ZeroQTrainConfig(compress_between_steps=False, quant_type="nf4")
        assert config.training_mode is True


@pytest.mark.skipif(not HAS_COORDINATOR, reason="coordinator not available (needs bitsandbytes)")
class TestTrainableParamLifecycle:
    """Tests for training-mode parameter lifecycle."""

    @staticmethod
    def _device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_partition_creates_master_shard(self):
        """partition() creates master_shard when training_mode=True."""
        config = ZeroQTrainConfig()
        model = torch.nn.Linear(64, 32, bias=False).to(self._device())
        coord = ZeroQCoordinator(config)
        zq = coord.register_parameter(model.weight, model, param_name="weight")
        zq.partition()

        assert zq.master_shard is not None
        expected_chunk = (64 * 32 + 1 - 1) // 1
        assert zq.master_shard.numel() == expected_chunk

    def test_gather_restores_shape(self):
        """start_gather reconstructs the original shape."""
        dev = self._device()
        config = ZeroQTrainConfig()
        model = torch.nn.Linear(64, 32, bias=False).to(dev)
        original = model.weight.data.clone()
        coord = ZeroQCoordinator(config)
        zq = coord.register_parameter(model.weight, model, param_name="weight")
        zq.partition()
        zq.start_gather(async_op=False)

        assert zq.param.shape == original.shape
        assert torch.allclose(zq.param.data, original.to(zq.param.device), atol=1e-6)

    def test_release_keeps_master_shard(self):
        """release() frees gathered param but preserves master_shard."""
        config = ZeroQTrainConfig()
        model = torch.nn.Linear(64, 32, bias=False).to(self._device())
        coord = ZeroQCoordinator(config)
        zq = coord.register_parameter(model.weight, model, param_name="weight")
        zq.partition()
        shard_data = zq.master_shard.data.clone()
        zq.start_gather(async_op=False)
        zq.release()

        assert zq.master_shard is not None
        assert torch.equal(zq.master_shard.data, shard_data)
        assert model.weight.numel() == 0

    def test_trainable_master_params(self):
        """Coordinator returns master shards for optimizer."""
        config = ZeroQTrainConfig()
        model = torch.nn.Linear(64, 32).to(self._device())
        coord = ZeroQCoordinator(config)
        wrapper = ZeroQModuleWrapper(model, coord)
        wrapper.partition()

        params = coord.trainable_master_params()
        assert len(params) == 2  # weight + bias

    def test_forward_backward_cycle(self):
        """Full forward-backward cycle with training mode hooks."""
        dev = self._device()
        config = ZeroQTrainConfig()
        model = torch.nn.Linear(64, 32, bias=False).to(dev)
        coord = ZeroQCoordinator(config)
        wrapper = ZeroQModuleWrapper(model, coord)
        wrapper.partition()

        master_params = coord.trainable_master_params()
        optimizer = torch.optim.SGD(master_params, lr=0.01)

        x = torch.randn(2, 64, device=dev)
        output = model(x)
        loss = output.sum()
        loss.backward()

        for p in master_params:
            assert p.grad is not None

        old_shard = master_params[0].data.clone()
        optimizer.step()
        assert not torch.equal(master_params[0].data, old_shard)
        optimizer.zero_grad()

    def test_checkpoint_roundtrip(self):
        """Gathered state matches original model weights."""
        dev = self._device()
        config = ZeroQTrainConfig()
        model = torch.nn.Linear(64, 32, bias=False).to(dev)
        original = model.weight.data.clone()
        coord = ZeroQCoordinator(config)
        zq = coord.register_parameter(model.weight, model, param_name="weight")
        zq.partition()

        zq.start_gather(async_op=False)
        gathered = zq.param.data.clone()
        zq.release()

        assert torch.allclose(gathered, original.to(gathered.device), atol=1e-6)

    def test_multi_step_training(self):
        """Loss decreases over multiple training steps on fixed data."""
        dev = self._device()
        config = ZeroQTrainConfig()
        torch.manual_seed(42)
        model = torch.nn.Linear(32, 16, bias=True).to(dev)
        coord = ZeroQCoordinator(config)
        wrapper = ZeroQModuleWrapper(model, coord)
        wrapper.partition()

        master_params = coord.trainable_master_params()
        optimizer = torch.optim.AdamW(master_params, lr=1e-2)

        torch.manual_seed(0)
        x = torch.randn(8, 32, device=dev)
        target = torch.randn(8, 16, device=dev)

        losses = []
        for _ in range(10):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_local_memory_bytes(self):
        """local_memory_bytes reflects master shard size."""
        config = ZeroQTrainConfig()
        model = torch.nn.Linear(64, 32, bias=False).to(self._device())
        coord = ZeroQCoordinator(config)
        zq = coord.register_parameter(model.weight, model, param_name="weight")
        zq.partition()

        expected = 64 * 32 * 4  # fp32, 4 bytes each
        assert zq.local_memory_bytes == expected
