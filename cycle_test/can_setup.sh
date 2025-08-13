#!/bin/bash

# Exit on any error
set -e

echo "🔧 Enabling CAN kernel modules..."
sudo modprobe can
sudo modprobe can-raw

echo "🛠️ Setting up CAN interface (can0)..."
sudo ip link set can0 type can bitrate 1000000 loopback off
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up

echo "📦 Installing Python dependencies..."

echo "✅ Setup complete. Run 'ip link' to verify CAN interface status."
