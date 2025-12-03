echo "Setting up Python 3.10 virtual environment..."
uv venv -p 3.10
source .venv/bin/activate
uv pip install pip
uv pip install -e .
uv pip install -r requirements.txt
uv pip install -U "nemo_toolkit[asr]"
echo "Parakeet ASR will auto-download from Hugging Face (no token needed)."
echo "For offline use, optionally download locally with:"
echo "  hf download nvidia/parakeet-tdt-0.6b-v2 --local-dir data/parakeet-tdt-0.6b-v2-hf"
echo "Then run with overrides: model.asr_model_id=data/parakeet-tdt-0.6b-v2-hf model.asr_local_files_only=True"

echo "Virtual environment setup complete. Please add allDatasets.pkl to the data/ directory. Run 'python src/train.py experiment=example' to start training."
