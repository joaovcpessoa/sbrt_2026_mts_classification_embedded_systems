"""Regra de negócio: split_mode='well' nunca pode colocar o mesmo poço em treino e teste.

Contexto: a auditoria de data leakage (.claude/HISTORY.md, "Auditoria de data leakage")
confirmou que split_mode='event' agrupa por arquivo, não por poço físico, e que isso
causa vazamento de entidade (195/195 eventos de teste pertenciam a poços já vistos no
treino, no dataset real). Estes testes usam WindowedData sintético -- não tocam
./data/ nem o dataset 3W real.
"""
import numpy as np
import pytest

from experiment import DataProcessor, PipelineConfig, WindowedData


def _make_data_processor(seed: int = 0) -> DataProcessor:
    cfg = PipelineConfig(random_seed=seed, train_ratio=0.5)
    return DataProcessor(cfg)


def _synthetic_data(rows_per_event: int = 10) -> WindowedData:
    """6 eventos sintéticos: 3 do poço W1, 2 do W2, 1 do W3 -- cenário em que o mesmo
    poço aparece em múltiplos eventos, exatamente o padrão do 3W real (42 poços para
    2228 eventos).
    """
    event_wells = {
        0: "WELL-00001", 1: "WELL-00001", 2: "WELL-00001",
        3: "WELL-00002", 4: "WELL-00002",
        5: "WELL-00003",
    }
    n_events = len(event_wells)
    n = n_events * rows_per_event

    X = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    y = np.resize(np.array([0, 1, 2], dtype=np.int64), n)
    groups = np.repeat(np.arange(n_events, dtype=np.int64), rows_per_event)
    wells = np.array([event_wells[int(g)] for g in groups], dtype=object)

    return WindowedData(X=X, y=y, groups=groups, wells=wells)


@pytest.mark.parametrize("seed", range(20))
def test_well_split_never_puts_same_well_on_both_sides(seed):
    data = _synthetic_data()
    dp = _make_data_processor(seed)
    rng = np.random.default_rng(seed)

    train_idx, test_idx = dp._split_by_key(data.wells, len(data.y), rng)

    assert len(train_idx) + len(test_idx) == len(data.y)
    train_wells = set(data.wells[train_idx])
    test_wells = set(data.wells[test_idx])
    assert not (train_wells & test_wells), (
        f"poço(s) {train_wells & test_wells} apareceram em treino e teste com seed={seed}"
    )


def test_event_split_can_leak_wells_while_well_split_never_does():
    """Documenta o bug original: agrupar por evento (não por poço) pode colocar o
    mesmo poço nos dois lados quando um poço tem vários eventos. Varre várias
    sementes porque, para uma única semente, o vazamento depende da ordem do sorteio.
    """
    data = _synthetic_data()
    dp = _make_data_processor()

    event_mode_leaked_at_least_once = False
    for seed in range(50):
        train_idx, _ = dp._split_by_key(data.groups, len(data.y), np.random.default_rng(seed))
        test_idx = np.setdiff1d(np.arange(len(data.y)), train_idx)
        train_wells = set(data.wells[train_idx])
        test_wells = set(data.wells[test_idx])
        if train_wells & test_wells:
            event_mode_leaked_at_least_once = True

        train_idx_w, test_idx_w = dp._split_by_key(data.wells, len(data.y), np.random.default_rng(seed))
        assert not (set(data.wells[train_idx_w]) & set(data.wells[test_idx_w]))

    assert event_mode_leaked_at_least_once, (
        "esperado que, com um poço de múltiplos eventos (W1), split por evento vaze o "
        "poço em pelo menos uma das 50 sementes testadas -- se isso não ocorre mais, o "
        "cenário sintético parou de representar o bug original"
    )


def test_entity_id_groups_real_wells_but_not_synthetic_instances():
    real_a = DataProcessor._entity_id("0/WELL-00001_20170201010101.parquet", "real")
    real_b = DataProcessor._entity_id("3/WELL-00001_20170305020000.parquet", "real")
    assert real_a == real_b == "WELL-00001"

    real_other_well = DataProcessor._entity_id("4/WELL-00002_20170101000000.parquet", "real")
    assert real_other_well != real_a

    sim_a = DataProcessor._entity_id("3/SIMULATED_00031.parquet", "simulated")
    sim_b = DataProcessor._entity_id("4/SIMULATED_00032.parquet", "simulated")
    assert sim_a != sim_b, "instâncias sintéticas não devem ser agrupadas na mesma entidade"
