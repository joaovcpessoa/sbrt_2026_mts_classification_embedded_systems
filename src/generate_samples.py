"""Gera os lotes de teste (e os sketches prontos) para rodar no Wokwi.

Uso:
    python generate_samples.py [--out DIR]

Treina UMA vez, quantiza (INT8/16/32) com o mesmo modelo e gera, para cada largura:
  - test_batches_int{bits}.py      : lotes de amostras quantizadas + rótulos + predição esperada
  - sketches/mlp_3w_int{bits}_batchNN.ino : sketch completo por lote (cole direto no Wokwi)
  - quant_int{bits}.npz            : pesos/escalas usados (rastreabilidade)

Rótulos: índices 0..K-1 (0 = Normal, 1 = Severe Slugging, 2 = Flow Instability, ver PipelineConfig).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from experiment import (
    PipelineConfig,
    DataProcessor,
    ModelManager,
    Quantizer,
    ArduinoExporter,
    set_global_seed,
    quantize_input,
)

N_BATCHES     = 10
SAMPLES_BATCH = 10

logger = logging.getLogger('generate_test_batches')

_C_TYPE = {8: 'int8_t', 16: 'int16_t', 32: 'int32_t'}


def build_config(out_dir: Path, samples_batch: int = SAMPLES_BATCH) -> PipelineConfig:
    return PipelineConfig(
        dataset_path=r'C:\Users\joaov_zm1q2wh\projects\sbrt_2026_mts_classification_embedded_systems\data',
        window_size=1000,
        random_seed=42,
        epochs=100,
        quantization_bits=[8, 16, 32],
        n_export=samples_batch,
        output_dir=out_dir,
        split_mode='well', # event
    )


def write_batches_file(batches: list[dict], bits: int, samples_batch: int, class_names: list[str], out_path: Path) -> None:
    """Escreve ``test_batches_int{bits}.py`` (dados) + blocos C comentados para colar no sketch."""
    c_type = _C_TYPE[bits]
    lines: list[str] = [
        f'BITS = {bits}',
        f'N_BATCHES = {len(batches)}',
        f'SAMPLES_PER_BATCH = {samples_batch}',
        f'LABELS = {dict(enumerate(class_names))}',
        '',
        'BATCHES = [',
    ]
    for b in batches:
        lines += [
            '    {',
            f"        'batch_id':      {b['batch_id']},",
            f"        'class_dist':    {b['class_dist']},",
            "        'X_test_q': [",
        ]
        lines += [f'            {row},' for row in b['X_test_q']]
        lines += [
            '        ],',
            f"        'y_labels':      {b['y_labels']},",
            f"        'expected_pred': {b['expected_pred']},   # emulacao inteira em Python",
            '    },',
            '',
        ]
    lines += [']', '', '']

    lines += ['# ' + '-' * 66, '# BLOCOS C (alternativa: use os sketches completos gerados)', '# ' + '-' * 66, '']
    for b in batches:
        flat_X = [v for row in b['X_test_q'] for v in row]
        lines += [
            f'# BATCH {b["batch_id"]:02d}  |  classes: {b["class_dist"]}',
            f'# const {c_type} X_test_q[] PROGMEM = {{',
        ]
        for i in range(0, len(flat_X), 16):
            lines.append('#   ' + ', '.join(map(str, flat_X[i:i + 16])) + ',')
        lines += [
            '# };',
            f'# const uint8_t y_labels[] PROGMEM = {{ {", ".join(map(str, b["y_labels"]))} }};',
            '',
        ]

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Arquivo gerado: %s', out_path)


def evaluate_fp32_pool(quantizer: Quantizer, mlp, X_pool: np.ndarray, y_pool: np.ndarray,
                       class_names: list[str]) -> None:
    layers = quantizer._get_all_linear_layers(mlp)
    _, logits = quantizer.forward_float(layers, X_pool)
    preds = logits.argmax(axis=1)
    n = len(y_pool)
    logger.info('-- FP32 no pool de %d amostras --', n)
    logger.info('  Acurácia: %d/%d = %.1f%%', int((preds == y_pool).sum()), n, (preds == y_pool).mean() * 100)
    for c in np.unique(y_pool):
        idx = y_pool == c
        logger.info('  %s: %d/%d', class_names[int(c)], int((preds[idx] == c).sum()), int(idx.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('./output'), help='pasta de saída')
    ap.add_argument('--n-batches', type=int, default=N_BATCHES, help='número de lotes/sketches por largura de bits')
    ap.add_argument('--samples-batch', type=int, default=SAMPLES_BATCH, help='amostras por lote/sketch')
    args = ap.parse_args()
    n_batches, samples_batch = args.n_batches, args.samples_batch

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)-8s  %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    cfg = build_config(args.out, samples_batch)
    cfg.sketches_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.random_seed)

    logger.info('Carregando dataset e aplicando windowing...')
    dp = DataProcessor(cfg)
    X_train, y_train, X_test, y_test = dp.split(dp.load_and_window())
    X_train, X_test = dp.normalize(X_train, X_test)

    logger.info('Treinando modelo (uma única vez)...')
    mgr = ModelManager(cfg)
    mgr.build_trainer()
    mgr.train(X_train, y_train)
    model = mgr.model
    model.eval()

    quantizer = Quantizer(cfg)
    exporter = ArduinoExporter(cfg)

    n_total = n_batches * samples_batch
    X_pool, y_pool = ArduinoExporter.select_balanced_export(
        X_test, y_test, n=n_total, seed=cfg.random_seed + 1
    )
    logger.info('Pool de %d amostras. Distribuição: %s', len(y_pool),
                {int(c): int((y_pool == c).sum()) for c in np.unique(y_pool)})
    if len(y_pool) < n_total:
        raise SystemExit(f'Conjunto de teste tem só {len(y_pool)} amostras; reduza --n-batches/--samples-batch.')

    evaluate_fp32_pool(quantizer, model, X_pool, y_pool, cfg.class_names)

    for bits in cfg.quantization_bits:
        logger.info('Processando INT%d...', bits)
        q = quantizer.quantize_model(model, X_train, bits)
        quantizer.save(q, cfg.output_dir / f'quant_int{bits}.npz')

        expected_all = quantizer.int_predict(q, X_pool)
        logger.info('INT%d no pool: %d/%d corretas (emulação inteira)', bits,
                    int((expected_all == y_pool).sum()), len(y_pool))

        batches: list[dict] = []
        for k in range(n_batches):
            sl = slice(k * samples_batch, (k + 1) * samples_batch)
            Xb, yb, eb = X_pool[sl], y_pool[sl], expected_all[sl]
            batches.append({
                'batch_id':      k,
                'X_test_q':      quantize_input(Xb, q['s_x'], bits).tolist(),
                'y_labels':      yb.astype(int).tolist(),
                'expected_pred': eb.astype(int).tolist(),
                'class_dist':    {int(c): int((yb == c).sum()) for c in np.unique(yb)},
            })
            exporter.export_sketch(q, Xb, yb, expected_pred=eb, suffix=f'_batch{k:02d}', n=samples_batch)

        write_batches_file(batches, bits, samples_batch, cfg.class_names, cfg.output_dir / f'test_batches_int{bits}.py')

    logger.info('Concluído. Saída em %s', cfg.output_dir.resolve())


if __name__ == '__main__':
    main()
