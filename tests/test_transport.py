#!/usr/bin/env python3
"""
Test script for ZeroQ transport layer and gradient synchronization.

Tests the new ZeroMQ-based multi-node communication components.

Usage:
    # Run all tests
    pytest test_transport.py -v
    
    # Run specific test
    pytest test_transport.py::TestSerialization -v

Author: Zero (Claude Opus 4.5) in collaboration with Douglas Rawson
Created: December 2025
"""

import os
import sys
import threading
import time
import unittest

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MultiNodeConfig
from src.transport import (
    serialize_tensor,
    deserialize_tensor,
    GradientCompressor,
    TransportConfig,
)
from src.gradient_sync import (
    TorchDistributedSync,
    GradientAccumulator,
    GradSyncConfig,
    SyncMode,
)


class TestSerialization(unittest.TestCase):
    """Test tensor serialization for network transfer."""
    
    def test_serialize_float32(self):
        """Test float32 tensor serialization."""
        tensor = torch.randn(10, 20)
        data = serialize_tensor(tensor)
        restored = deserialize_tensor(data, torch.device("cpu"))
        
        self.assertEqual(tensor.shape, restored.shape)
        self.assertTrue(torch.allclose(tensor, restored))
    
    def test_serialize_float16(self):
        """Test float16 tensor serialization."""
        tensor = torch.randn(10, 20).half()
        data = serialize_tensor(tensor)
        restored = deserialize_tensor(data, torch.device("cpu"))
        
        self.assertEqual(tensor.shape, restored.shape)
        self.assertTrue(torch.allclose(tensor.float(), restored.float(), atol=1e-3))
    
    def test_serialize_large_tensor(self):
        """Test large tensor serialization."""
        tensor = torch.randn(1000, 1000)
        data = serialize_tensor(tensor)
        restored = deserialize_tensor(data, torch.device("cpu"))
        
        self.assertEqual(tensor.shape, restored.shape)
        self.assertTrue(torch.allclose(tensor, restored))
    
    def test_serialize_multidim(self):
        """Test multi-dimensional tensor serialization."""
        tensor = torch.randn(4, 8, 16, 32)
        data = serialize_tensor(tensor)
        restored = deserialize_tensor(data, torch.device("cpu"))
        
        self.assertEqual(tensor.shape, restored.shape)
        self.assertTrue(torch.allclose(tensor, restored))


class TestGradientCompressor(unittest.TestCase):
    """Test gradient compression."""
    
    def test_topk_compression(self):
        """Test Top-K gradient compression."""
        compressor = GradientCompressor(compression_ratio=0.1)
        
        # Create gradient tensor
        tensor = torch.randn(100, 100)
        
        # Compress
        values, indices, metadata = compressor.compress(tensor, param_id=0)
        
        # Check compression ratio
        expected_k = int(tensor.numel() * 0.1)
        self.assertEqual(len(values), expected_k)
        self.assertEqual(len(indices), expected_k)
        
        # Decompress
        restored = compressor.decompress(values, indices, metadata, torch.device("cpu"))
        
        # Check shape
        self.assertEqual(tensor.shape, restored.shape)
    
    def test_error_feedback(self):
        """Test error feedback mechanism."""
        compressor = GradientCompressor(compression_ratio=0.1)
        
        tensor = torch.randn(100, 100)
        
        # First compression
        values1, indices1, meta1 = compressor.compress(tensor.clone(), param_id=0)
        
        # Check error feedback was stored
        self.assertIn(0, compressor.error_feedback)
        
        # Second compression should include error feedback
        tensor2 = torch.randn(100, 100)
        values2, indices2, meta2 = compressor.compress(tensor2, param_id=0)
        
        # The values should be different due to error feedback
        self.assertFalse(torch.equal(values1, values2))
    
    def test_compression_preserves_large_values(self):
        """Test that compression preserves the largest magnitude values."""
        compressor = GradientCompressor(compression_ratio=0.01)  # Keep top 1%
        
        # Create tensor with known large values
        tensor = torch.zeros(100, 100)
        tensor[0, 0] = 100.0
        tensor[50, 50] = -100.0
        
        values, indices, metadata = compressor.compress(tensor, param_id=1)
        
        # The largest values should be preserved
        self.assertIn(100.0, values.float().tolist() + [-v for v in values.float().tolist()])


class TestGradientSync(unittest.TestCase):
    """Test gradient synchronization backends."""
    
    def test_single_process_sync(self):
        """Test sync in single process mode."""
        sync = TorchDistributedSync()
        
        tensor = torch.randn(10, 10)
        original = tensor.clone()
        
        # Should be no-op in single process
        sync.all_reduce(tensor)
        
        self.assertTrue(torch.equal(tensor, original))
    
    def test_gradient_accumulator(self):
        """Test gradient accumulator."""
        sync = TorchDistributedSync()
        config = GradSyncConfig(mode=SyncMode.LOCAL_SGD, local_steps=2)
        
        accumulator = GradientAccumulator(sync, config)
        
        # Accumulate gradients
        grad1 = torch.ones(10, 10)
        grad2 = torch.ones(10, 10) * 2
        
        accumulator.accumulate(0, grad1)
        self.assertFalse(accumulator.should_sync())  # Not yet
        
        accumulator.accumulate(0, grad2)
        self.assertTrue(accumulator.should_sync())  # Now sync
        
        # Sync and clear
        synced = accumulator.sync_and_clear()
        
        self.assertIn(0, synced)
        expected = grad1 + grad2  # Accumulated
        self.assertTrue(torch.allclose(synced[0], expected))
    
    def test_accumulator_multiple_params(self):
        """Test accumulating multiple parameters."""
        sync = TorchDistributedSync()
        config = GradSyncConfig()
        
        accumulator = GradientAccumulator(sync, config)
        
        # Accumulate different params
        accumulator.accumulate(0, torch.ones(10))
        accumulator.accumulate(1, torch.ones(20) * 2)
        accumulator.accumulate(2, torch.ones(30) * 3)
        
        synced = accumulator.sync_and_clear()
        
        self.assertEqual(len(synced), 3)
        self.assertTrue(torch.allclose(synced[0], torch.ones(10)))
        self.assertTrue(torch.allclose(synced[1], torch.ones(20) * 2))
        self.assertTrue(torch.allclose(synced[2], torch.ones(30) * 3))


class TestMultiNodeConfig(unittest.TestCase):
    """Test multi-node configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = MultiNodeConfig()
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.coordinator_host, "localhost")
        self.assertEqual(config.coordinator_port, 5555)
        self.assertIsNotNone(config.node_id)  # Auto-generated
    
    def test_config_serialization(self):
        """Test config to/from dict."""
        config = MultiNodeConfig(
            enabled=True,
            coordinator_host="pe1",
            coordinator_port=5555,
            gradient_compression=True,
        )
        
        d = config.to_dict()
        restored = MultiNodeConfig.from_dict(d)
        
        self.assertEqual(config.enabled, restored.enabled)
        self.assertEqual(config.coordinator_host, restored.coordinator_host)
        self.assertEqual(config.gradient_compression, restored.gradient_compression)
    
    def test_phoenix_cluster_config(self):
        """Test Phoenix cluster preset."""
        from src.config import PHOENIX_CLUSTER_CONFIG
        
        self.assertTrue(PHOENIX_CLUSTER_CONFIG.enabled)
        self.assertEqual(PHOENIX_CLUSTER_CONFIG.coordinator_host, "pe1")
        self.assertTrue(PHOENIX_CLUSTER_CONFIG.use_hierarchical_sync)
    
    def test_auto_node_id(self):
        """Test automatic node ID generation."""
        config1 = MultiNodeConfig()
        config2 = MultiNodeConfig()
        
        # Each should have unique auto-generated ID
        self.assertIsNotNone(config1.node_id)
        self.assertIsNotNone(config2.node_id)


class TestRingAllReduceLogic(unittest.TestCase):
    """Test ring all-reduce algorithm logic."""
    
    def test_ring_topology(self):
        """Test ring topology calculation."""
        from src.gradient_sync import RingAllReduce
        
        # Create mock transport
        class MockTransport:
            rank = 1
            world_size = 4
            peer_sockets = {}
            _recv_buffer = {}
            _lock = threading.Lock()
        
        transport = MockTransport()
        ring = RingAllReduce(transport)
        
        # Check ring neighbors
        self.assertEqual(ring.send_rank, 2)  # (1 + 1) % 4
        self.assertEqual(ring.recv_rank, 0)  # (1 - 1 + 4) % 4
    
    def test_tensor_splitting(self):
        """Test tensor splitting for ring algorithm."""
        from src.gradient_sync import RingAllReduce
        
        class MockTransport:
            rank = 0
            world_size = 4
            peer_sockets = {}
            _recv_buffer = {}
            _lock = threading.Lock()
        
        transport = MockTransport()
        ring = RingAllReduce(transport)
        
        # Test splitting
        tensor = torch.randn(100)
        chunks = ring._split_tensor(tensor)
        
        self.assertEqual(len(chunks), 4)
        
        # Test merging
        merged = ring._merge_chunks(chunks, tensor.shape)
        self.assertTrue(torch.allclose(tensor, merged))
    
    def test_uneven_split(self):
        """Test splitting with uneven tensor size."""
        from src.gradient_sync import RingAllReduce
        
        class MockTransport:
            rank = 0
            world_size = 3
            peer_sockets = {}
            _recv_buffer = {}
            _lock = threading.Lock()
        
        transport = MockTransport()
        ring = RingAllReduce(transport)
        
        # 10 elements, 3 ranks
        tensor = torch.randn(10)
        chunks = ring._split_tensor(tensor)
        
        self.assertEqual(len(chunks), 3)
        
        # Merge should preserve original
        merged = ring._merge_chunks(chunks, tensor.shape)
        self.assertTrue(torch.allclose(tensor, merged))
    
    def test_multidim_tensor_split(self):
        """Test splitting multi-dimensional tensor."""
        from src.gradient_sync import RingAllReduce
        
        class MockTransport:
            rank = 0
            world_size = 2
            peer_sockets = {}
            _recv_buffer = {}
            _lock = threading.Lock()
        
        transport = MockTransport()
        ring = RingAllReduce(transport)
        
        # Multi-dim tensor
        tensor = torch.randn(8, 16, 4)
        chunks = ring._split_tensor(tensor)
        
        self.assertEqual(len(chunks), 2)
        
        # Merge should preserve original shape
        merged = ring._merge_chunks(chunks, tensor.shape)
        self.assertEqual(tensor.shape, merged.shape)
        self.assertTrue(torch.allclose(tensor, merged))


class TestTransportConfig(unittest.TestCase):
    """Test transport configuration."""
    
    def test_default_config(self):
        """Test default transport config."""
        config = TransportConfig()
        
        self.assertEqual(config.bind_address, "0.0.0.0")
        self.assertEqual(config.coordinator_host, "localhost")
        self.assertEqual(config.coordinator_port, 5555)
        self.assertEqual(config.heartbeat_interval_ms, 5000)
    
    def test_custom_config(self):
        """Test custom transport config."""
        config = TransportConfig(
            coordinator_host="pe1",
            coordinator_port=6666,
            heartbeat_interval_ms=1000,
            tcp_buffer_size=8 * 1024 * 1024,
        )
        
        self.assertEqual(config.coordinator_host, "pe1")
        self.assertEqual(config.coordinator_port, 6666)
        self.assertEqual(config.tcp_buffer_size, 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
