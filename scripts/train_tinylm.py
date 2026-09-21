"""WSLのリポジトリルートから実行する学習入口。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aster.training.pretrain import main

if __name__ == "__main__":
    main()
