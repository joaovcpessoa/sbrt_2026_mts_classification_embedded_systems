from __future__ import annotations

import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (
    PipelineConfig,
    DataProcessor,
    ModelManager,
    Quantizer,
    ArduinoExporter,
    set_global_seed,
    _NP_WEIGHT
)

CFG = PipelineConfig(
    dataset_path=r'C:\Users\jvt\Downloads\data\3W',
    window_size=1000,
    random_seed=42,
    epochs=100,
    quantization_bits=[8, 16, 32],
    n_export=10,
    output_dir=Path('./output'),
)

N_BATCHES     = 10
SAMPLES_BATCH = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('generate_test_batches')


def quantize_samples(X_batch: np.ndarray, s_x: float, bits: int) -> np.ndarray:
    clip_max = 2 ** (bits - 1) - 1
    clip_min = -(2 ** (bits - 1))
    return np.clip(np.round(X_batch / s_x), clip_min, clip_max).astype(_NP_WEIGHT[bits])


def write_batches_file(batches: list[dict], bits: int, out_path: Path) -> None:
    c_type_map = {8: 'int8_t', 16: 'int16_t', 32: 'int32_t'}
    c_type = c_type_map[bits]

    lines: list[str] = [
        f'BITS = {bits}',
        f'N_BATCHES = {len(batches)}',
        f'SAMPLES_PER_BATCH = {SAMPLES_BATCH}',
        '',
        'BATCHES = [',
    ]

    for b in batches:
        bid  = b['batch_id']
        X_q  = b['X_test_q']
        y    = b['y_labels']
        dist = b['class_dist']

        lines += [
            f'    {{',
            f'        \'batch_id\':   {bid},',
            f'        \'class_dist\': {dist},',
            f'        \'X_test_q\': [',
        ]
        for row in X_q:
            lines.append(f'            {row},')
        lines += [
            f'        ],',
            f'        \'y_labels\': {y},',
            f'    }},',
            '',
        ]

    lines += [']', '', '']

    lines += [
        '# ' + '─' * 66,
        '# BLOCOS C PRONTOS PARA COLAR NO SKETCH ARDUINO',
        '# ' + '─' * 66,
        '',
    ]

    for b in batches:
        bid    = b['batch_id']
        X_q    = b['X_test_q']
        y      = b['y_labels']
        dist   = b['class_dist']
        flat_X = [v for row in X_q for v in row]

        lines += [
            f'# {"=" * 64}',
            f'# BATCH {bid:02d}  |  classes: {dist}',
            f'# {"=" * 64}',
            f'# const {c_type} X_test_q[] PROGMEM = {{',
        ]
        for i in range(0, len(flat_X), 16):
            chunk = ', '.join(map(str, flat_X[i:i + 16]))
            lines.append(f'#   {chunk},')
        lines += [
            '# };',
            '#',
            f'# const uint8_t y_labels[] PROGMEM = {{',
            f'#   {", ".join(map(str, y))}',
            '# };',
            '',
        ]

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Arquivo gerado: %s', out_path)


def evaluate_fp32_pool(model: torch.nn.Module, X_pool: np.ndarray, y_pool: np.ndarray) -> None:
    device = next(model.parameters()).device
    X_t = torch.tensor(X_pool, dtype=torch.float32).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(X_t)
    elapsed = time.perf_counter() - t0

    preds = logits.argmax(dim=1).cpu().numpy()
    correct = int((preds == y_pool).sum())
    n = len(y_pool)
    acc = correct / n

    avg_time_us = elapsed / n * 1e6
    classes = np.unique(y_pool)
    class_names = {0: 'Normal Operation', 3: 'Flow Instability', 4: 'Severe Slugging'}

    logger.info('── FP32 evaluation on the %d-sample pool ──', n)
    logger.info('  Accuracy        : %d/%d = %.1f%%', correct, n, acc * 100)
    logger.info('  Total time      : %.3f s', elapsed)
    logger.info('  Avg time/sample : %.2f us', avg_time_us)
    for c in classes:
        idx = y_pool == c
        c_correct = int((preds[idx] == y_pool[idx]).sum())
        logger.info('  %s: %d/%d = %.1f%%', class_names.get(int(c), str(c)), c_correct, idx.sum(), c_correct / idx.sum() * 100)

def main() -> None:
    set_global_seed(CFG.random_seed)

    logger.info('Carregando dataset e aplicando windowing...')
    data_proc = DataProcessor(CFG)
    df = data_proc.load_and_window()
    X_train, y_train, X_test, y_test = data_proc.split(df)

    logger.info('Treinando modelo...')
    mgr = ModelManager(CFG)
    mgr.build_trainer()
    mgr.train(X_train, y_train)
    model = mgr.model
    model.eval()

    quantizer = Quantizer(CFG)

    total_needed = N_BATCHES * SAMPLES_BATCH
    X_pool, y_pool = ArduinoExporter.select_balanced_export(
        X_test, y_test, n=total_needed, seed=CFG.random_seed + 1
    )

    classes = np.unique(y_pool)
    logger.info(
        'Pool de %d amostras selecionado. Distribuição: %s',
        total_needed,
        {int(c): int((y_pool == c).sum()) for c in classes},
    )

    evaluate_fp32_pool(model, X_pool, y_pool)

    for bits in CFG.quantization_bits:
        logger.info('Processando INT%d...', bits)
        q = quantizer.quantize_model(model, X_train, bits)

        batches: list[dict] = []
        for batch_id in range(N_BATCHES):
            start   = batch_id * SAMPLES_BATCH
            end     = start + SAMPLES_BATCH
            X_batch = X_pool[start:end]
            y_batch = y_pool[start:end]
            X_batch_q = quantize_samples(X_batch, q['s_x'], bits)
            dist = {int(c): int((y_batch == c).sum()) for c in np.unique(y_batch)}

            batches.append({
                'batch_id':   batch_id,
                'X_test_q':   X_batch_q.tolist(),
                'y_labels':   y_batch.astype(int).tolist(),
                'class_dist': dist,
            })
            logger.info('  Batch %02d | classes: %s', batch_id, dist)

        write_batches_file(batches, bits, Path(f'test_batches_int{bits}.py'))

    logger.info('Concluido! Arquivos gerados:')
    for bits in CFG.quantization_bits:
        logger.info('  test_batches_int%d.py', bits)


main()