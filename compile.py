"""Compile eigenprinters.cpp into a library file"""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run():
    result = subprocess.run(['g++', '-g', '-c', SCRIPT_DIR / 'eigenprinters.cpp'])
    if result.returncode == 0:
        print('Done! Eigenprinters library compiled')
    else:
        print(f'Compilation failed with code {result.returncode}\n{result.stderr}', file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == '__main__':
    run()
