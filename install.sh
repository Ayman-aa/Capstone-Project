#!/bin/bash
# ============================================================
# STEP 1 — Install Ollama and pull all three models
# Run this once before anything else.
# Usage: bash step1_setup.sh
# ============================================================

set -e

echo ""
echo "============================================"
echo " The Cost of Intelligence — Experiment Setup"
echo "============================================"
echo ""

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null; then
    echo "[1/4] Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
            echo "      Ollama installed."
            else
                echo "[1/4] Ollama already installed. Skipping."
                fi

                # Start Ollama service in background if not running
                echo "[2/4] Starting Ollama service..."
                if ! pgrep -x "ollama" > /dev/null; then
                    ollama serve &> /tmp/ollama.log &
                        sleep 3
                            echo "      Ollama service started."
                            else
                                echo "      Ollama service already running."
                                fi

                                # Pull the three models
                                echo ""
                                echo "[3/4] Pulling models (this will take a while on first run)..."
                                echo ""

                                echo "  --> Pulling TinyLlama (1.1B) ..."
                                ollama pull tinyllama
                                echo "      Done."

                                echo ""
                                echo "  --> Pulling Phi-3 Mini (3.8B) ..."
                                ollama pull phi3:mini
                                echo "      Done."

                                echo ""
                                echo "  --> Pulling Gemma 2 (9B) ..."
                                ollama pull gemma2:9b
                                echo "      Done."

                                echo ""
                                echo "[4/4] Verifying models are available..."
                                ollama list

                                echo ""
                                echo "============================================"
                                echo " Setup complete. You can now run:"
                                echo " python3 step2_run_experiment.py"
                                echo "============================================"
                                echo ""