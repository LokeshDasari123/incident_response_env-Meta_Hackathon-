#!/bin/bash
url='http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces'

mkdir -p data/raw/Node
mkdir -p data/raw/MSResource
mkdir -p data/raw/MSRTQps
mkdir -p data/raw/MSCallGraph

echo "Downloading Node_0 (~90MB)..."
wget -c --retry-connrefused --tries=3 --timeout=50 \
  /node/Node_0.tar.gz \
  -O data/raw/Node/Node_0.tar.gz

echo "Downloading MSResource_0 (~1.3GB)..."
wget -c --retry-connrefused --tries=3 --timeout=50 \
  /MSResource/MSResource_0.tar.gz \
  -O data/raw/MSResource/MSResource_0.tar.gz

echo "Downloading MSRTQps_0 (~800MB)..."
wget -c --retry-connrefused --tries=3 --timeout=50 \
  /MSRTQps/MSRTQps_0.tar.gz \
  -O data/raw/MSRTQps/MSRTQps_0.tar.gz

echo "Downloading MSCallGraph_0 (~170MB)..."
wget -c --retry-connrefused --tries=3 --timeout=50 \
  /MSCallGraph/MSCallGraph_0.tar.gz \
  -O data/raw/MSCallGraph/MSCallGraph_0.tar.gz

echo "Extracting all..."
cd data/raw/Node && tar -xzf Node_0.tar.gz && cd ../../..
cd data/raw/MSResource && tar -xzf MSResource_0.tar.gz && cd ../../..
cd data/raw/MSRTQps && tar -xzf MSRTQps_0.tar.gz && cd ../../..
cd data/raw/MSCallGraph && tar -xzf MSCallGraph_0.tar.gz && cd ../../..

echo "Done. Total ~2.4GB downloaded and extracted."
