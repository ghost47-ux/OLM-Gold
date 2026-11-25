import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from olm.config import ModelConfig
from olm.models.olm import build_model

for dm, nl, nh, dff, cl, drop in [(192,4,6,512,2,0.10),(160,4,5,448,2,0.05),(128,4,4,384,2,0.05),(128,3,4,384,1,0.05),(96,3,4,384,1,0.05)]:
    c = ModelConfig(d_model=dm, n_layers=nl, n_heads=nh, d_ff=dff, context_encoder_layers=cl, dropout=drop)
    m = build_model(c, 14)
    print(dm, nl, nh, dff, cl, drop, m.parameter_count())
