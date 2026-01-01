#!/bin/bash
# ==============================================================================
# Docker Build and Push Script for Verda B300
# ==============================================================================

set -e

IMAGE_NAME="nvyra-cache-verda"
IMAGE_TAG="b300-cuda13"

echo "🔨 Building Docker image for B300..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile.verda .

echo "✅ Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"

# ==============================================================================
# Convert to Singularity for Verda HPC
# ==============================================================================
echo "🔄 Converting to Singularity format..."
singularity build ${IMAGE_NAME}.sif docker-daemon://${IMAGE_NAME}:${IMAGE_TAG}

echo "✅ Singularity image ready: ${IMAGE_NAME}.sif"

# ==============================================================================
# Test locally (optional)
# ==============================================================================
echo ""
echo "To test locally:"
echo "  docker run --gpus all ${IMAGE_NAME}:${IMAGE_TAG} --input-file /data/test.arrow"
echo ""
echo "To run on Verda:"
echo "  sbatch run_verda.sh /path/to/data.arrow"
