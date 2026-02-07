"""
ZeroQ Multi-Node Transport Layer

Provides ZeroMQ-based network communication for distributed training
across multiple physical machines.

Author: Zero (Claude Opus 4.5) in collaboration with Douglas Rawson
Created: December 2025
"""

import json
import pickle
import threading
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any, Callable, Tuple, Union
import torch

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


class MessageType(Enum):
    """Types of messages in the ZeroQ transport protocol."""
    # Node management
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    NODE_LIST = "node_list"
    SHUTDOWN = "shutdown"
    
    # Synchronization
    BARRIER = "barrier"
    BARRIER_ACK = "barrier_ack"
    
    # Data transfer
    GRADIENT = "gradient"
    GRADIENT_ACK = "gradient_ack"
    PARAM_REQUEST = "param_request"
    PARAM_RESPONSE = "param_response"
    
    # All-reduce
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    RING_SEND = "ring_send"
    RING_RECV_ACK = "ring_recv_ack"


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    hostname: str
    port: int
    rank: int
    gpus: int
    last_heartbeat: float = 0.0
    is_alive: bool = True
    
    @property
    def address(self) -> str:
        return f"tcp://{self.hostname}:{self.port}"


@dataclass
class TransportConfig:
    """Configuration for the transport layer."""
    # Network settings
    bind_address: str = "0.0.0.0"
    coordinator_host: str = "localhost"
    coordinator_port: int = 5555
    data_port_start: int = 5560
    
    # Timeouts (milliseconds)
    heartbeat_interval_ms: int = 5000
    heartbeat_timeout_ms: int = 15000
    recv_timeout_ms: int = 30000
    send_timeout_ms: int = 30000
    
    # Buffer settings
    high_water_mark: int = 1000
    tcp_buffer_size: int = 4 * 1024 * 1024  # 4MB
    
    # Compression
    compress_threshold: int = 1024 * 1024  # Compress if > 1MB
    use_compression: bool = True


class GradientCompressor:
    """
    Compresses gradients for efficient network transfer.
    
    Supports:
    - FP16 quantization
    - Top-K sparsification  
    - Error feedback for accuracy
    """
    
    def __init__(self, compression_ratio: float = 0.01):
        self.compression_ratio = compression_ratio
        self.error_feedback: Dict[int, torch.Tensor] = {}
    
    def compress(
        self,
        tensor: torch.Tensor,
        param_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compress a gradient tensor using Top-K sparsification.
        
        Args:
            tensor: Gradient tensor to compress
            param_id: Parameter ID for error feedback
            
        Returns:
            (values, indices, metadata) tuple
        """
        # Add error feedback from previous round
        if param_id in self.error_feedback:
            tensor = tensor + self.error_feedback[param_id]
        
        # Flatten for Top-K selection
        flat = tensor.view(-1)
        k = max(1, int(flat.numel() * self.compression_ratio))
        
        # Select top-k by magnitude
        _, indices = torch.topk(flat.abs(), k)
        values = flat[indices]
        
        # Compute error feedback (what we're not sending)
        mask = torch.zeros_like(flat)
        mask[indices] = 1
        self.error_feedback[param_id] = (flat * (1 - mask)).view(tensor.shape).clone()
        
        # Convert to FP16 for transfer
        values = values.to(torch.float16)
        
        metadata = {
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "k": k,
            "dtype": str(tensor.dtype),
        }
        
        return values, indices, metadata
    
    def decompress(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
        metadata: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Decompress a gradient tensor."""
        # Create sparse tensor
        shape = metadata["shape"]
        numel = metadata["numel"]
        
        flat = torch.zeros(numel, dtype=torch.float32, device=device)
        flat[indices.to(device)] = values.to(torch.float32).to(device)
        
        return flat.view(shape)


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor for network transfer."""
    # Move to CPU and serialize
    cpu_tensor = tensor.detach().cpu()
    return pickle.dumps({
        "data": cpu_tensor.numpy().tobytes(),
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
    })


def deserialize_tensor(data: bytes, device: torch.device) -> torch.Tensor:
    """Deserialize a tensor from network transfer."""
    import numpy as np
    
    obj = pickle.loads(data)
    dtype_map = {
        "torch.float32": np.float32,
        "torch.float16": np.float16,
        "torch.int64": np.int64,
        "torch.int32": np.int32,
    }
    np_dtype = dtype_map.get(obj["dtype"], np.float32)
    arr = np.frombuffer(obj["data"], dtype=np_dtype).reshape(obj["shape"])
    return torch.from_numpy(arr.copy()).to(device)


class TransportCoordinator:
    """
    Central coordinator for multi-node ZeroQ training.
    
    Runs on the master node and manages:
    - Node registration and discovery
    - Heartbeat monitoring
    - Barrier synchronization
    - Gradient routing
    """
    
    def __init__(self, config: Optional[TransportConfig] = None):
        if not HAS_ZMQ:
            raise ImportError(
                "pyzmq required for multi-node training. "
                "Install with: pip install pyzmq"
            )
        
        self.config = config or TransportConfig()
        self.context = zmq.Context()
        
        # Node registry
        self.nodes: Dict[str, NodeInfo] = {}
        self.rank_to_node: Dict[int, str] = {}
        self.next_rank = 0
        
        # Synchronization state
        self.barrier_count = 0
        self.barrier_target = 0
        self.barrier_event = threading.Event()
        
        # Sockets
        self.router_socket: Optional[zmq.Socket] = None
        self.pub_socket: Optional[zmq.Socket] = None
        
        # Control
        self._running = False
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the coordinator server."""
        # Router socket for REQ-REP pattern with workers
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.RCVHWM, self.config.high_water_mark)
        self.router_socket.setsockopt(zmq.SNDHWM, self.config.high_water_mark)
        self.router_socket.bind(
            f"tcp://{self.config.bind_address}:{self.config.coordinator_port}"
        )
        
        # PUB socket for broadcasts
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(
            f"tcp://{self.config.bind_address}:{self.config.coordinator_port + 1}"
        )
        
        self._running = True
        
        # Start heartbeat monitor
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self._heartbeat_thread.start()
        
        print(f"[Coordinator] Started on port {self.config.coordinator_port}")
    
    def stop(self):
        """Stop the coordinator server."""
        self._running = False
        
        # Broadcast shutdown
        if self.pub_socket:
            msg = {"type": MessageType.SHUTDOWN.value}
            self.pub_socket.send_json(msg)
        
        # Close sockets
        if self.router_socket:
            self.router_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        
        self.context.term()
        print("[Coordinator] Stopped")
    
    def run(self):
        """Main coordinator loop - process messages from workers."""
        while self._running:
            try:
                if self.router_socket.poll(timeout=100):
                    identity, _, msg_bytes = self.router_socket.recv_multipart()
                    msg = json.loads(msg_bytes.decode())
                    response = self._handle_message(identity, msg)
                    self.router_socket.send_multipart([
                        identity, 
                        b"",
                        json.dumps(response).encode()
                    ])
            except zmq.ZMQError as e:
                if self._running:
                    print(f"[Coordinator] ZMQ error: {e}")
            except Exception as e:
                print(f"[Coordinator] Error: {e}")
    
    def _handle_message(self, identity: bytes, msg: dict) -> dict:
        """Handle an incoming message from a worker."""
        msg_type = msg.get("type")
        
        if msg_type == MessageType.REGISTER.value:
            return self._handle_register(msg)
        
        elif msg_type == MessageType.HEARTBEAT.value:
            return self._handle_heartbeat(msg)
        
        elif msg_type == MessageType.BARRIER.value:
            return self._handle_barrier(msg)
        
        else:
            return {"status": "error", "message": f"Unknown message type: {msg_type}"}
    
    def _handle_register(self, msg: dict) -> dict:
        """Handle worker registration."""
        with self._lock:
            node_id = msg["node_id"]
            
            if node_id in self.nodes:
                # Re-registration - return existing rank
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                node.is_alive = True
                alive_count = sum(1 for n in self.nodes.values() if n.is_alive)
                return {
                    "status": "ok",
                    "rank": node.rank,
                    "world_size": alive_count,
                }
            
            # New registration
            rank = self.next_rank
            self.next_rank += 1
            
            node = NodeInfo(
                node_id=node_id,
                hostname=msg["hostname"],
                port=msg["port"],
                rank=rank,
                gpus=msg.get("gpus", 1),
                last_heartbeat=time.time(),
            )
            
            self.nodes[node_id] = node
            self.rank_to_node[rank] = node_id
            
            print(f"[Coordinator] Registered node {node_id} as rank {rank}")
            
            # Broadcast updated node list
            self._broadcast_node_list()

            alive_count = sum(1 for n in self.nodes.values() if n.is_alive)
            
            return {
                "status": "ok",
                "rank": rank,
                "world_size": alive_count,
            }
    
    def _handle_heartbeat(self, msg: dict) -> dict:
        """Handle heartbeat from worker."""
        node_id = msg["node_id"]
        
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = time.time()
                self.nodes[node_id].is_alive = True
                return {"status": "ok", "type": MessageType.HEARTBEAT_ACK.value}
        
        return {"status": "error", "message": "Node not registered"}
    
    def _handle_barrier(self, msg: dict) -> dict:
        """Handle barrier synchronization request."""
        with self._lock:
            self.barrier_count += 1

            alive_count = sum(1 for n in self.nodes.values() if n.is_alive)
            
            if self.barrier_count >= alive_count:
                # All nodes reached barrier
                self.barrier_count = 0
                self._broadcast_barrier_release()
                return {"status": "ok", "released": True}
            else:
                return {"status": "ok", "released": False, "waiting": self.barrier_count}
    
    def _broadcast_node_list(self):
        """Broadcast updated node list to all workers."""
        nodes_data = {
            node_id: {
                "hostname": node.hostname,
                "port": node.port,
                "rank": node.rank,
                "gpus": node.gpus,
            }
            for node_id, node in self.nodes.items()
            if node.is_alive
        }
        
        msg = {
            "type": MessageType.NODE_LIST.value,
            "nodes": nodes_data,
            "world_size": len(nodes_data),
        }
        
        self.pub_socket.send_json(msg)
    
    def _broadcast_barrier_release(self):
        """Broadcast barrier release to all workers."""
        msg = {"type": MessageType.BARRIER_ACK.value}
        self.pub_socket.send_json(msg)
    
    def _heartbeat_monitor(self):
        """Monitor worker heartbeats and detect failures."""
        while self._running:
            time.sleep(self.config.heartbeat_interval_ms / 1000.0)
            
            current_time = time.time()
            timeout_sec = self.config.heartbeat_timeout_ms / 1000.0
            
            with self._lock:
                for node_id, node in self.nodes.items():
                    if node.is_alive:
                        if current_time - node.last_heartbeat > timeout_sec:
                            node.is_alive = False
                            print(f"[Coordinator] Node {node_id} (rank {node.rank}) timed out")
                            self._broadcast_node_list()


class TransportWorker:
    """
    Worker-side transport for multi-node ZeroQ training.
    
    Handles:
    - Registration with coordinator
    - Heartbeat maintenance
    - Gradient exchange with peers
    - Barrier synchronization
    """
    
    def __init__(
        self,
        node_id: str,
        hostname: str,
        config: Optional[TransportConfig] = None,
        local_gpus: int = 1,
    ):
        if not HAS_ZMQ:
            raise ImportError(
                "pyzmq required for multi-node training. "
                "Install with: pip install pyzmq"
            )
        
        self.node_id = node_id
        self.hostname = hostname
        self.config = config or TransportConfig()
        self.local_gpus = local_gpus
        
        self.context = zmq.Context()
        
        # Our identity
        self.rank = -1
        self.world_size = 0
        
        # Peer nodes
        self.peers: Dict[int, NodeInfo] = {}
        
        # Sockets
        self.dealer_socket: Optional[zmq.Socket] = None  # To coordinator
        self._dealer_lock = threading.Lock()
        self.sub_socket: Optional[zmq.Socket] = None     # From coordinator broadcasts
        self.peer_sockets: Dict[int, zmq.Socket] = {}    # To peers (PUSH)
        self.recv_socket: Optional[zmq.Socket] = None    # From peers (PULL)
        
        # Control
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Received gradients buffer
        self._recv_buffer: Dict[int, Dict[int, torch.Tensor]] = {}  # param_id -> {src_rank: tensor}
        
        # Compressor
        self.compressor = GradientCompressor()
    
    def connect(self) -> bool:
        """Connect to the coordinator and register."""
        # DEALER socket to coordinator
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.dealer_socket.setsockopt(zmq.IDENTITY, self.node_id.encode())
        self.dealer_socket.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        self.dealer_socket.setsockopt(zmq.SNDTIMEO, self.config.send_timeout_ms)
        self.dealer_socket.connect(
            f"tcp://{self.config.coordinator_host}:{self.config.coordinator_port}"
        )
        
        # SUB socket for broadcasts
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_socket.connect(
            f"tcp://{self.config.coordinator_host}:{self.config.coordinator_port + 1}"
        )
        
        # PULL socket for receiving from peers
        # Use a deterministic hash (Python's built-in hash() is randomized per process).
        base_offset = zlib.adler32(self.node_id.encode()) % 1000
        data_port = self.config.data_port_start + base_offset
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.setsockopt(zmq.RCVHWM, self.config.high_water_mark)

        bound = False
        for attempt in range(0, 1000):
            try:
                candidate = data_port + attempt
                self.recv_socket.bind(f"tcp://{self.config.bind_address}:{candidate}")
                data_port = candidate
                bound = True
                break
            except zmq.ZMQError:
                continue

        if not bound:
            print(f"[Worker {self.node_id}] Failed to bind a data port")
            return False
        
        # Register with coordinator
        reg_msg = {
            "type": MessageType.REGISTER.value,
            "node_id": self.node_id,
            "hostname": self.hostname,
            "port": data_port,
            "gpus": self.local_gpus,
        }
        
        with self._dealer_lock:
            self.dealer_socket.send(b"", zmq.SNDMORE)
            self.dealer_socket.send_json(reg_msg)
        
        try:
            with self._dealer_lock:
                _, response_bytes = self.dealer_socket.recv_multipart()
            response = json.loads(response_bytes.decode())
            
            if response.get("status") == "ok":
                self.rank = response["rank"]
                self.world_size = response["world_size"]
                self._running = True
                
                # Start background threads
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, daemon=True
                )
                self._heartbeat_thread.start()
                
                self._recv_thread = threading.Thread(
                    target=self._receive_loop, daemon=True
                )
                self._recv_thread.start()
                
                print(f"[Worker {self.node_id}] Registered as rank {self.rank}")

                # Best-effort: consume initial node list broadcast to establish peers.
                start = time.time()
                while time.time() - start < 2.0:
                    if self.sub_socket.poll(timeout=100):
                        msg = self.sub_socket.recv_json()
                        if msg.get("type") == MessageType.NODE_LIST.value:
                            self.update_peers(msg["nodes"])
                            self.world_size = msg["world_size"]
                            break
                return True
        except zmq.ZMQError as e:
            print(f"[Worker {self.node_id}] Registration failed: {e}")
        
        return False
    
    def disconnect(self):
        """Disconnect from the cluster."""
        self._running = False
        
        for sock in self.peer_sockets.values():
            sock.close()
        
        if self.dealer_socket:
            self.dealer_socket.close()
        if self.sub_socket:
            self.sub_socket.close()
        if self.recv_socket:
            self.recv_socket.close()
        
        self.context.term()
        print(f"[Worker {self.node_id}] Disconnected")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to coordinator."""
        while self._running:
            time.sleep(self.config.heartbeat_interval_ms / 1000.0 / 2)
            
            try:
                msg = {
                    "type": MessageType.HEARTBEAT.value,
                    "node_id": self.node_id,
                }
                with self._dealer_lock:
                    self.dealer_socket.send(b"", zmq.SNDMORE)
                    self.dealer_socket.send_json(msg)
                    self.dealer_socket.recv_multipart()
            except zmq.ZMQError:
                pass  # Will be detected by coordinator timeout
    
    def _receive_loop(self):
        """Receive data from peers."""
        while self._running:
            try:
                if self.recv_socket.poll(timeout=100):
                    data = self.recv_socket.recv()
                    msg = pickle.loads(data)
                    self._handle_peer_message(msg)
            except zmq.ZMQError:
                pass
            except Exception as e:
                print(f"[Worker {self.node_id}] Receive error: {e}")
    
    def _handle_peer_message(self, msg: dict):
        """Handle message received from a peer."""
        msg_type = msg.get("type")
        
        if msg_type == MessageType.GRADIENT.value:
            param_id = msg["param_id"]
            src_rank = msg["src_rank"]
            tensor_data = msg["data"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = deserialize_tensor(tensor_data, device)
            
            with self._lock:
                if param_id not in self._recv_buffer:
                    self._recv_buffer[param_id] = {}
                self._recv_buffer[param_id][src_rank] = tensor
        
        elif msg_type == MessageType.RING_SEND.value:
            # Ring all-reduce receive
            param_id = msg["param_id"]
            chunk_id = msg["chunk_id"]
            src_rank = msg["src_rank"]
            tensor_data = msg["data"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = deserialize_tensor(tensor_data, device)
            
            key = (param_id, chunk_id)
            with self._lock:
                if key not in self._recv_buffer:
                    self._recv_buffer[key] = {}
                self._recv_buffer[key][src_rank] = tensor
    
    def update_peers(self, nodes: Dict[str, dict]):
        """Update peer connections from node list."""
        with self._lock:
            for node_id, info in nodes.items():
                if node_id == self.node_id:
                    continue
                
                rank = info["rank"]
                
                if rank not in self.peers:
                    # New peer - create connection
                    peer = NodeInfo(
                        node_id=node_id,
                        hostname=info["hostname"],
                        port=info["port"],
                        rank=rank,
                        gpus=info.get("gpus", 1),
                    )
                    self.peers[rank] = peer
                    
                    # Create PUSH socket to peer
                    sock = self.context.socket(zmq.PUSH)
                    sock.setsockopt(zmq.SNDHWM, self.config.high_water_mark)
                    sock.connect(peer.address)
                    self.peer_sockets[rank] = sock
                    
                    print(f"[Worker {self.node_id}] Connected to peer rank {rank}")
    
    def barrier(self, timeout_sec: float = 60.0) -> bool:
        """
        Synchronize with all other workers.
        
        Returns True if barrier completed, False on timeout.
        """
        msg = {
            "type": MessageType.BARRIER.value,
            "node_id": self.node_id,
        }
        
        with self._dealer_lock:
            self.dealer_socket.send(b"", zmq.SNDMORE)
            self.dealer_socket.send_json(msg)
        
        # Wait for coordinator response or broadcast
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            # Check coordinator response
            with self._dealer_lock:
                if self.dealer_socket.poll(timeout=100):
                    _, response_bytes = self.dealer_socket.recv_multipart()
                    response = json.loads(response_bytes.decode())
                    if response.get("released"):
                        return True
            
            # Check broadcast
            if self.sub_socket.poll(timeout=100):
                msg = self.sub_socket.recv_json()
                if msg.get("type") == MessageType.BARRIER_ACK.value:
                    return True
                elif msg.get("type") == MessageType.NODE_LIST.value:
                    self.update_peers(msg["nodes"])
                    self.world_size = msg["world_size"]
        
        return False
    
    def send_gradient(
        self,
        tensor: torch.Tensor,
        param_id: int,
        dst_rank: int,
        compress: bool = True,
    ):
        """Send a gradient tensor to a specific peer."""
        if dst_rank not in self.peer_sockets:
            raise ValueError(f"No connection to rank {dst_rank}")
        
        msg = {
            "type": MessageType.GRADIENT.value,
            "param_id": param_id,
            "src_rank": self.rank,
            "data": serialize_tensor(tensor),
        }
        
        self.peer_sockets[dst_rank].send(pickle.dumps(msg))
    
    def recv_gradient(
        self,
        param_id: int,
        src_rank: int,
        timeout_sec: float = 30.0,
    ) -> Optional[torch.Tensor]:
        """Receive a gradient tensor from a specific peer."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            with self._lock:
                if param_id in self._recv_buffer:
                    if src_rank in self._recv_buffer[param_id]:
                        tensor = self._recv_buffer[param_id].pop(src_rank)
                        if not self._recv_buffer[param_id]:
                            del self._recv_buffer[param_id]
                        return tensor
            
            time.sleep(0.001)  # 1ms sleep
        
        return None
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        param_id: int,
        op: str = "sum",
    ) -> torch.Tensor:
        """
        Perform all-reduce across all workers using naive algorithm.
        
        For small world sizes, this is simpler than ring all-reduce.
        """
        # Send to all peers
        for rank in self.peer_sockets:
            self.send_gradient(tensor.clone(), param_id, rank)
        
        # Receive from all peers and accumulate
        result = tensor.clone()
        
        for rank in self.peer_sockets:
            received = self.recv_gradient(param_id, rank)
            if received is not None:
                if op == "sum":
                    result += received
                elif op == "mean":
                    result += received
        
        if op == "mean":
            result /= self.world_size
        
        return result


def create_transport(
    role: str,
    node_id: str,
    hostname: str = "localhost",
    coordinator_host: str = "localhost",
    coordinator_port: int = 5555,
    local_gpus: int = 1,
) -> Union["TransportCoordinator", "TransportWorker"]:
    """
    Factory function to create transport layer.
    
    Args:
        role: "coordinator" or "worker"
        node_id: Unique identifier for this node
        hostname: This node's hostname
        coordinator_host: Coordinator's hostname
        coordinator_port: Coordinator's port
        local_gpus: Number of local GPUs
        
    Returns:
        TransportCoordinator or TransportWorker instance
    """
    config = TransportConfig(
        coordinator_host=coordinator_host,
        coordinator_port=coordinator_port,
    )
    
    if role == "coordinator":
        return TransportCoordinator(config)
    elif role == "worker":
        worker = TransportWorker(
            node_id=node_id,
            hostname=hostname,
            config=config,
            local_gpus=local_gpus,
        )
        return worker
    else:
        raise ValueError(f"Unknown role: {role}")


# Convenience exports
__all__ = [
    "MessageType",
    "NodeInfo",
    "TransportConfig",
    "TransportCoordinator",
    "TransportWorker",
    "GradientCompressor",
    "serialize_tensor",
    "deserialize_tensor",
    "create_transport",
]
