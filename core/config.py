from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

@dataclass
class ModelConfig:
    device: str = 'cpu'
    savepath: str = 'results/'
    assets_dir: Path = ASSETS_DIR
    iscrop: bool = True
    detector: str = 'retinaface'
    ldm68: bool = True
    ldm106: bool = True
    ldm106_2d: bool = True
    ldm134: bool = True
    seg_visible: bool = True
    seg: bool = True
    useTex: bool = True
    extractTex: bool = True
    backbone: str = 'resnet50'