#!/usr/bin/env python3
"""
ZeroQ Simple Network Test - PE1 <-> PE3

Tests basic ZeroMQ communication between PE1 and PE3 without
the complexity of full gradient sync.

Run on PE1:  python simple_network_test.py --role server
Run on PE3:  python simple_network_test.py --role client --server-host poweredge1
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transport import serialize_tensor, deserialize_tensor

try:
    import zmq
except ImportError:
    print("ERROR: pyzmq not installed. Run: pip install pyzmq")
    sys.exit(1)


def run_server(port=5556):
    """Run as server (on PE1)."""
    print("=" * 50)
    print("ZeroQ Network Test - SERVER (PE1)")
    print("=" * 50)
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    print(f"Server listening on port {port}")
    print("Waiting for client...")
    
    # Receive tensor from client
    msg = socket.recv()
    tensor = deserialize_tensor(msg, torch.device("cpu"))
    print(f"Received tensor: shape={tensor.shape}, sum={tensor.sum().item():.4f}")
    
    # Send back modified tensor
    response = tensor * 2
    socket.send(serialize_tensor(response))
    print(f"Sent response: shape={response.shape}, sum={response.sum().item():.4f}")
    
    # Second round - with GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\nGPU test on {torch.cuda.get_device_name(0)}...")
        
        msg = socket.recv()
        tensor = deserialize_tensor(msg, device)
        print(f"Received on GPU: shape={tensor.shape}, device={tensor.device}")
        
        # Do computation on GPU
        result = tensor @ tensor.T  # Matrix multiply
        result = result.cpu()  # Move back to CPU for sending
        socket.send(serialize_tensor(result))
        print(f"Sent GPU result: shape={result.shape}")
    
    socket.close()
    context.term()
    print("\n✓ Server test complete!")


def run_client(server_host, port=5556):
    """Run as client (on PE3)."""
    print("=" * 50)
    print("ZeroQ Network Test - CLIENT (PE3)")
    print("=" * 50)
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_host}:{port}")
    
    print(f"Connected to server at {server_host}:{port}")
    
    # Create and send tensor
    tensor = torch.randn(100, 100)
    print(f"Sending tensor: shape={tensor.shape}, sum={tensor.sum().item():.4f}")
    socket.send(serialize_tensor(tensor))
    
    # Receive response
    msg = socket.recv()
    response = deserialize_tensor(msg, torch.device("cpu"))
    print(f"Received response: shape={response.shape}, sum={response.sum().item():.4f}")
    
    # Verify
    expected_sum = tensor.sum().item() * 2
    actual_sum = response.sum().item()
    if abs(expected_sum - actual_sum) < 0.01:
        print("✓ Response verified!")
    else:
        print(f"✗ Mismatch: expected {expected_sum:.4f}, got {actual_sum:.4f}")
    
    # Second round - with GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\nGPU test on {torch.cuda.get_device_name(0)}...")
        
        tensor = torch.randn(50, 50, device=device)
        tensor_cpu = tensor.cpu()
        print(f"Sending from GPU: shape={tensor.shape}")
        socket.send(serialize_tensor(tensor_cpu))
        
        msg = socket.recv()
        result = deserialize_tensor(msg, device)
        print(f"Received GPU result: shape={result.shape}")
        
        # Verify matrix multiply
        expected = tensor_cpu @ tensor_cpu.T
        if torch.allclose(expected, result.cpu(), atol=1e-5):
            print("✓ GPU computation verified!")
        else:
            print("✗ GPU computation mismatch")
    
    socket.close()
    context.term()
    print("\n✓ Client test complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--server-host", default="poweredge1")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()
    
    if args.role == "server":
        run_server(args.port)
    else:
        run_client(args.server_host, args.port)


if __name__ == "__main__":
    main()
