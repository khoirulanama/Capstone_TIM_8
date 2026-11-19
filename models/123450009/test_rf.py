import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_rf as b

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 400, 6)).astype("float32")
f = b.fitur(X)
print(f.shape)   