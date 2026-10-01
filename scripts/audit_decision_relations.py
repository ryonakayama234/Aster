"""Replay and independently audit every saved relation-study prediction."""
import argparse
import json
from pathlib import Path
from audit_decision_mixed_v1 import audit
from aster.training.decision_relations import SERIALIZERS

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.run,serializers={n:s for n,(_,s) in SERIALIZERS.items()}),sort_keys=True))
