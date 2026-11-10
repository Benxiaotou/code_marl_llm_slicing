"""Run existing contracts in isolated processes, with no external model access."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ('tests', 'test_*.py'),
    ('gpt61_adapter_tests', 'test_*.py'),
    ('p2d_replay', 'test*.py'),
    ('p2d_node_link', 'test*.py'),
    ('p2e_prompt/tests', 'test_*.py'),
    ('p2f_analysis', 'test*.py'),
    ('p2h_v_decomposition', 'test*.py'),
    ('p2i_node_boundary', 'test*.py'),
    ('p2j_validation', 'test*.py'),
    ('p2l_cached_test', 'test*.py'),
    ('p2m_paper', 'test*.py'),
    ('p3_gpt_heterogeneous_dev', 'test*.py'),
)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.upper() in ('LLM_API_KEY', 'LLM_BASE_URL', 'GPT61_API_KEY', 'GPT61_BASE_URL',
                            'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_API_KEY'):
            environment.pop(name)
    environment['PYTHONPATH'] = str(ROOT / 'tools/offline_guard') + os.pathsep + str(ROOT)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONIOENCODING'] = 'utf-8'
    environment['NO_PROXY'] = 'localhost,127.0.0.1,::1'
    environment['WANDB_MODE'] = 'disabled'
    # Keep temporary fixtures inside the checkout for restricted workspaces.
    temporary_root = ROOT / '.offline_test_tmp'
    temporary_root.mkdir(exist_ok=True)
    environment['TEMP'] = str(temporary_root)
    environment['TMP'] = str(temporary_root)
    results = []
    for directory, pattern in SUITES:
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', directory, '-p', pattern, '-v']
        process = subprocess.run(command, cwd=str(ROOT), env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = process.stdout.decode('utf-8', 'replace').replace('\r\n', '\n').replace('\r', '')
        match = re.search(r'Ran (\d+) tests? in ', output)
        skipped = re.search(r'skipped=(\d+)', output)
        result = {'suite': directory, 'tests': int(match.group(1)) if match else 0,
                  'skipped': int(skipped.group(1)) if skipped else 0,
                  'exit_code': process.returncode, 'passed': process.returncode == 0 and match is not None}
        results.append(result)
        print(json.dumps(result), flush=True)
        if not result['passed']:
            start = output.find('\n======================================================================')
            print(output[start:start+8000] if start >= 0 else output[-2000:], flush=True)
    report = {'python': sys.version.split()[0], 'suites': results,
              'tests': sum(x['tests'] for x in results), 'passed': all(x['passed'] for x in results),
              'skipped': sum(x['skipped'] for x in results),
              'external_network_blocked': True, 'real_model_credentials_removed': True,
              'new_model_calls': 0, 'additional_model_api_cost_usd': 0}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'total_tests': report['tests'], 'passed': report['passed']}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
