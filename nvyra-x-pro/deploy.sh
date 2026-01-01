#!/bin/bash
# Deploy NVYRA-X PRO to Modal

echo "🚀 Deploying NVYRA-X PRO..."
modal deploy inference.py

echo "✅ Deployment complete!"
echo "📡 Endpoint: https://nvyra-x-pro--verify-claim.modal.run"
