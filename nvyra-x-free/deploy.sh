#!/bin/bash
# Deploy NVYRA-X FREE to Modal

echo "🚀 Deploying NVYRA-X FREE..."
modal deploy inference.py

echo "✅ Deployment complete!"
echo "📡 Endpoint: https://nvyra-x-free--verify-claim.modal.run"
