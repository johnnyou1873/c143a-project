echo "Setting up Python 3.10 virtual environment..."
uv venv -p 3.10
source .venv/bin/activate
uv pip install pip
uv pip install -e .
uv pip install -r requirements.txt
echo "Virtual environment setup complete. Please add allDatasets.pkl to the data/ directory. Run 'python src/train.py experiment=example' to start training."