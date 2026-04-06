"""scripts/process_alibaba.py — Run this once after downloading Alibaba data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from data.synthetic.generator import main
if __name__ == "__main__":
    main()