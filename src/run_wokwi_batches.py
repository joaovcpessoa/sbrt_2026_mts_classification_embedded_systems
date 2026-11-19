"""
Roda, em loop e via linha de comando, todos os lotes .ino gerados por generate_samples.py 
no simulador Wokwi via wokwi-cli + PlatformIO, sem precisar colar nada manualmente na IDE web

Pré-requisitos locais:
1. PlatformIO: pip install platformio
2. wokwi-cli: veja https://docs.wokwi.com/wokwi-ci/cli-installation (Windows: iwr https://wokwi.com/ci/install.ps1 -useb | iex)
3. Wokwi Token

Uso:
- python generate_samples.py --out output --n-batches 100 --samples-batch 10
- python run_wokwi_batches.py --sketches-dir output/sketches
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_WOKWI_DIR = HERE.parent / 'wokwi_ci'

ACC_RE = re.compile(r'Acuracia Final:\s*(\d+)/(\d+)')
DIV_RE = re.compile(r'Divergencia vs emulacao Python:\s*(\d+)/(\d+)')


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def build_firmware(wokwi_dir: Path, pio_env: str) -> None:
    r = run(['pio', 'run', '-e', pio_env], cwd=wokwi_dir, timeout=600)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f'Falha ao compilar com PlatformIO (env={pio_env}). Veja a saída acima.')


def run_wokwi(wokwi_dir: Path, timeout_ms: int, log_file: Path) -> tuple[bool, str]:
    r = run(
        [
            'wokwi-cli', str(wokwi_dir),
            '--timeout', str(timeout_ms),
            '--expect-text', 'Energia Total',
            '--fail-text', 'Guru Meditation',
            '--serial-log-file', str(log_file),
        ],
        timeout=timeout_ms / 1000 + 60,
    )
    ok = r.returncode == 0
    return ok, r.stdout + '\n' + r.stderr


def parse_results(log_file: Path) -> tuple[int, int, int, int] | None:
    """Retorna (corretas, total, divergentes, total) do log serial de um lote."""
    if not log_file.exists():
        return None
    text = log_file.read_text(encoding='utf-8', errors='replace')
    acc = ACC_RE.search(text)
    div = DIV_RE.search(text)
    if not acc:
        return None
    correct, n = int(acc.group(1)), int(acc.group(2))
    mismatch, n2 = (int(div.group(1)), int(div.group(2))) if div else (0, n)
    return correct, n, mismatch, n2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sketches-dir', type=Path, default=Path('output/sketches'))
    ap.add_argument('--wokwi-dir', type=Path, default=DEFAULT_WOKWI_DIR)
    ap.add_argument('--bits', type=int, nargs='+', default=[8, 16, 32])
    ap.add_argument('--timeout', type=int, default=10000, help='timeout por lote, em ms simulados')
    ap.add_argument('--pio-env', default='esp32dev')
    args = ap.parse_args()

    if shutil.which('pio') is None:
        raise SystemExit('PlatformIO não encontrado no PATH. Instale com: pip install platformio')
    if shutil.which('wokwi-cli') is None:
        raise SystemExit(
            'wokwi-cli não encontrado no PATH. Veja: https://docs.wokwi.com/wokwi-ci/cli-installation'
        )
    if not os.environ.get('WOKWI_CLI_TOKEN'):
        raise SystemExit('Defina a env var WOKWI_CLI_TOKEN (token gerado em https://wokwi.com/dashboard/ci).')

    main_ino = args.wokwi_dir / 'src' / 'main.ino'
    log_dir = args.wokwi_dir / '_run_logs'
    log_dir.mkdir(exist_ok=True)

    for bits in args.bits:
        batches = sorted(args.sketches_dir.glob(f'mlp_3w_int{bits}_batch*.ino'))
        if not batches:
            print(f'[INT{bits}] nenhum sketch encontrado em {args.sketches_dir} '
                  f'(rode generate_samples_v2.py primeiro)')
            continue

        total_correct = total_samples = total_mismatch = total_div_n = 0
        failed_batches: list[str] = []

        print(f'\n=== INT{bits}: {len(batches)} lotes ===')
        for sketch in batches:
            shutil.copy(sketch, main_ino)
            build_firmware(args.wokwi_dir, args.pio_env)

            log_file = log_dir / f'{sketch.stem}.log'
            ok, cli_output = run_wokwi(args.wokwi_dir, args.timeout, log_file)

            parsed = parse_results(log_file)
            if parsed is None:
                failed_batches.append(sketch.name)
                print(f'  {sketch.name}: FALHOU (sem saída válida; wokwi-cli ok={ok})')
                print('  ' + cli_output.strip().replace('\n', '\n  ')[-1000:])
                continue

            correct, n, mismatch, div_n = parsed
            total_correct += correct
            total_samples += n
            total_mismatch += mismatch
            total_div_n += div_n
            status = 'ok' if ok and mismatch == 0 else 'ATENCAO'
            print(f'  {sketch.name}: {correct}/{n} corretas | divergencia vs Python {mismatch}/{div_n} [{status}]')

        print(f'--- INT{bits} total: {total_correct}/{total_samples} corretas no ESP32 '
              f'({total_correct / max(total_samples, 1) * 100:.1f}%) | '
              f'divergencia vs emulacao Python: {total_mismatch}/{total_div_n}')
        if failed_batches:
            print(f'  lotes com falha: {failed_batches}')

    print('\nLogs completos de cada lote em', log_dir)


if __name__ == '__main__':
    main()
