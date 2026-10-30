"""OFFLINE ONLY: derive role-only historical IR/cost facts from selected MAPPO cases."""
import argparse
import json
from p1d.ir import build_ir_overlay


def main(argv=None):
    p = argparse.ArgumentParser(description='Build immutable training-only self-IR overlay for P1-D')
    p.add_argument('--library', required=True, help='Audited P1-B original library directory')
    p.add_argument('--output-dir', required=True, help='NEW separate IR overlay directory')
    args = p.parse_args(argv)
    meta = build_ir_overlay(args.library, args.output_dir)
    print(json.dumps({'ok':True, 'training_dataset_id':meta['training_dataset_id'],
                      'n_per_role':meta['n_selected_training_cases'],
                      'output_dir':args.output_dir}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
