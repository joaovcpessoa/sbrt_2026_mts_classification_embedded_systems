"""
Teste forense F-01: verifica vazamento de entidade (poço) entre treino e teste

Contexto: o 3W tem só dezenas de poços para milhares de eventos (ver
docs/presentations/ROTEIRO.md e .claude/HISTORY.md, "Auditoria de data leakage"), então
agrupar o split por evento (arquivo), em vez de por poço físico, tende a colocar o mesmo
poço em treino e teste -- o modelo pode aprender a assinatura do sensor de um poço
específico em vez do fenômeno físico da classe. `DataProcessor` já suporta os três modos
(`split_mode='well'|'event'|'window'`); este script reusa a lógica real do pipeline
(`DataProcessor.load_and_window`/`_split_by_key`) para medir a sobreposição de poços em
qualquer um dos três, em vez de reimplementar a extração/divisão à parte.

Uso (da raiz do projeto, mesmo ambiente do experiment.py):
    python src/check_well_leakage.py                      # split_mode='event' (default do Pipeline)
    python src/check_well_leakage.py --split-mode well     # modo corrigido: 0 sobreposição esperada
    python src/check_well_leakage.py --split-mode window   # modo legado, vazamento ainda pior

Saída esperada se NÃO houver vazamento de poço: "poços em ambos" = 0
"""
from __future__ import annotations

import argparse

import numpy as np

from experiment import PipelineConfig, DataProcessor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        '--split-mode', choices=['well', 'event', 'window'], default='event',
        help="modo de split a verificar (default: 'event', o default do Pipeline)",
    )
    ap.add_argument(
        '--dataset-path',
        default=r'C:\Users\joaov_zm1q2wh\projects\sbrt_2026_mts_classification_embedded_systems\data',
        help='caminho local do dataset 3W',
    )
    ap.add_argument('--seed', type=int, default=42, help='random_seed do PipelineConfig')
    ap.add_argument('--train-ratio', type=float, default=0.8, help='train_ratio do PipelineConfig')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        dataset_path=args.dataset_path,
        window_size=1000,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
    )

    dp = DataProcessor(cfg)
    data = dp.load_and_window()

    n_total = len(data.y)
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.split_mode == 'window':
        # mesma lógica de DataProcessor.split() para split_mode='window': sem grupo
        # nenhum, janela por janela -- reproduzida aqui porque _split_by_key não cobre
        # esse modo (ele existe só para 'event'/'well', que compartilham o algoritmo).
        idx = rng.permutation(n_total)
        n_train = int(cfg.train_ratio * n_total)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
    else:
        key = data.groups if cfg.split_mode == 'event' else data.wells
        # reusa a implementação real do pipeline (o mesmo método que
        # DataProcessor.split() chama por baixo), para medir exatamente o que o
        # treinamento de verdade produziria -- não uma reimplementação à parte.
        train_idx, test_idx = dp._split_by_key(key, n_total, rng)

    train_wells = set(data.wells[train_idx])
    test_wells = set(data.wells[test_idx])
    overlap = train_wells & test_wells

    print(f"split_mode={cfg.split_mode!r}  seed={cfg.random_seed}  train_ratio={cfg.train_ratio}")
    print(f'Janelas: {n_total} total  ({len(train_idx)} treino / {len(test_idx)} teste)')
    print(f'Poços distintos (total): {len(train_wells | test_wells)}')
    print(f'Poços só no treino: {len(train_wells - test_wells)}')
    print(f'Poços só no teste:  {len(test_wells - train_wells)}')
    print(f'Poços em AMBOS (vazamento de entidade, se > 0): {len(overlap)}')
    if overlap:
        shown = sorted(overlap)[:20]
        print(f"  IDs: {shown}{' ...' if len(overlap) > 20 else ''}")

    test_from_overlap = int(np.isin(data.wells[test_idx], list(overlap)).sum()) if overlap else 0
    print(
        f'Janelas de TESTE pertencentes a um poço também visto no treino: '
        f'{test_from_overlap} / {len(test_idx)}'
    )


if __name__ == '__main__':
    main()
