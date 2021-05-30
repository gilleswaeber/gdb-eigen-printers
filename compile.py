"""Compile the object file for eigenprinters.cpp"""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run():
    result = subprocess.run(['g++', '-g', '-c', SCRIPT_DIR / 'eigenprinters.cpp', '-o', SCRIPT_DIR / 'eigenprinters.o'])
    if result.returncode == 0:
        print('Done! Eigenprinters ready to be used')
    else:
        print(f'Compilation failed with code {result.returncode}\n{result.stderr}', file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == '__main__':
    run()
