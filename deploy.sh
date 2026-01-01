#!/bin/bash
# nvyra-x deployment and benchmark script

set -e

echo "nvyra-x sota deployment script"
echo "==============================="
echo ""

# load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# command selection
case "$1" in
    deploy-free)
        echo "deploying free tier (cpu) to modal..."
        cd nvyra-x-free
        modal deploy inference.py
        echo "done! endpoint deployed."
        ;;
    
    deploy-pro)
        echo "deploying pro tier (h200) to modal..."
        cd nvyra-x-pro
        modal deploy inference.py
        echo "done! endpoint deployed."
        ;;
    
    deploy-all)
        echo "deploying all tiers to modal..."
        cd nvyra-x-free && modal deploy inference.py && cd ..
        cd nvyra-x-pro && modal deploy inference.py && cd ..
        echo "done! all endpoints deployed."
        ;;
    
    run-free)
        echo "running free tier test..."
        cd nvyra-x-free
        modal run inference.py
        ;;
    
    run-pro)
        echo "running pro tier test..."
        cd nvyra-x-pro
        modal run inference.py
        ;;
    
    build-rust)
        echo "building rust client (release mode with lto)..."
        cd nvyra-x-rust
        cargo build --release
        echo "binary location: nvyra-x-rust/target/release/nvyra-x-client"
        ;;
    
    benchmark)
        echo "running rust client benchmark..."
        if [ -z "$2" ]; then
            echo "usage: $0 benchmark <input.csv>"
            exit 1
        fi
        
        cd nvyra-x-rust
        cargo run --release -- \
            --pro-url "${NVYRA_PRO_URL}" \
            --free-url "${NVYRA_FREE_URL}" \
            --input "$2" \
            --output "benchmark_results.jsonl" \
            --benchmark \
            --max-connections 500 \
            --batch-size 128 \
            --max-batch-size 512
        ;;
    
    help|*)
        echo "usage: $0 <command>"
        echo ""
        echo "commands:"
        echo "  deploy-free   deploy free tier to modal"
        echo "  deploy-pro    deploy pro tier to modal"
        echo "  deploy-all    deploy all tiers to modal"
        echo "  run-free      test free tier locally"
        echo "  run-pro       test pro tier locally"
        echo "  build-rust    build rust client (release)"
        echo "  benchmark     run throughput benchmark"
        echo ""
        ;;
esac
