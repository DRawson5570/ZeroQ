#!/bin/bash
# ZeRO-Q Test Runner
# Run all tests and generate a report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZEROQ_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "ZeRO-Q Test Suite"
echo "========================================"
echo "Date: $(date)"
echo "Directory: $ZEROQ_DIR"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Results tracking
PASSED=0
FAILED=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo -n "Running $test_name... "
    
    if output=$(eval "$test_cmd" 2>&1); then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "$output" | head -20
        ((FAILED++))
        return 1
    fi
}

# Change to ZeRO-Q directory
cd "$ZEROQ_DIR"

echo ""
echo "Unit Tests (CPU)"
echo "----------------------------------------"

run_test "test_config" "python -c 'from src.config import ZeroQConfig, MAXWELL_CONFIG; print(MAXWELL_CONFIG)'"
run_test "test_partition_imports" "python -c 'from src.partition import compute_aligned_partition_sizes, partition_quantized_tensor'"
run_test "test_coordinator_imports" "python -c 'from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper'"
run_test "test_integration_imports" "python -c 'from src.integration import initialize, prepare_model_for_zeroq'"
run_test "test_checkpoint_imports" "python -c 'from src.checkpoint import enable_gradient_checkpointing, estimate_checkpoint_memory'"

echo ""
echo "GPU Tests (if available)"
echo "----------------------------------------"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    run_test "test_gpu_quantization" "python -m pytest tests/test_gpu_quantization.py -v --tb=short" || true
    run_test "test_distributed_comm" "python -m pytest tests/test_distributed_comm.py -v --tb=short" || true
else
    echo -e "${YELLOW}SKIPPED: No CUDA available${NC}"
fi

echo ""
echo "Memory Calculator"
echo "----------------------------------------"
run_test "test_memory_calculator" "python tools/memory_calculator.py > /dev/null" || true

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
