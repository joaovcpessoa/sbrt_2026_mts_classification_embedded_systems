from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from codecarbon import EmissionsTracker

import torch
import torch.nn as nn
import torch.optim as optim

from ThreeWToolkit.dataset import ParquetDataset, ParquetDatasetConfig
from ThreeWToolkit.feature_extraction import Windowing, WindowingConfig
from ThreeWToolkit.models.mlp import MLP, MLPConfig
from ThreeWToolkit.trainer import TorchTrainer, TorchTrainerConfig
from ThreeWToolkit.assessment import (
    ModelAssessment,
    ModelAssessmentConfig,
    AssessmentVisualization,
    AssessmentVisualizationConfig,
)
from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from ThreeWToolkit.core.base_trainer import TrainingResult, TrainingHistory

# Mapeamentos config (string) -> objeto real exigido pelo ThreeWToolkit.
# MLPConfig.activation_function espera uma instância de nn.Module (não string);
# TorchTrainerConfig.optimizer/criterion esperam a CLASSE (não string).
_ACTIVATIONS: dict[str, nn.Module] = {
    'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(),
}
_OPTIMIZERS: dict[str, type] = {
    'adam': optim.Adam, 'adamw': optim.AdamW, 'sgd': optim.SGD, 'rmsprop': optim.RMSprop,
}
_CRITERIA: dict[str, type] = {
    'cross_entropy': nn.CrossEntropyLoss, 'mse': nn.MSELoss,
}

# Mapeamentos de tipos C/NumPy por largura de bits
_NP_WEIGHT: dict[int, type] = {8: np.int8,  16: np.int16, 32: np.int32}
_NP_BIAS:   dict[int, type] = {8: np.int32, 16: np.int32, 32: np.int64}
_C_WEIGHT:  dict[int, str]  = {8: 'int8_t',  16: 'int16_t', 32: 'int32_t'}
_C_BIAS:    dict[int, str]  = {8: 'int32_t', 16: 'int32_t', 32: 'int64_t'}
_C_ACCUM:   dict[int, str]  = {8: 'int32_t', 16: 'int64_t', 32: 'int64_t'}

# Limite (em bits, sem o sinal) do acumulador de cada largura: int32 -> 31, int64 -> 63
_ACC_BITS:  dict[int, int]  = {8: 31, 16: 63, 32: 63}

# Epsilon da Eq. de escala (absmax simétrico). Em INT32 ele domina a escala dos pesos
# (max|W| / 2^31 é da ordem de 1e-10), então INT32 usa menos de 32 bits efetivos.
_EPS = 1e-8

# Nomes das classes do 3W Dataset 2.0.0 (Tabela 4 do artigo do dataset)
CLASS_NAMES_3W: dict[int, str] = {
    0: 'Normal Operation',
    1: 'Abrupt Increase of BSW',
    2: 'Spurious Closure of DHSV',
    3: 'Severe Slugging',
    4: 'Flow Instability',
}


# Reproducibility
def set_global_seed(seed: int) -> None:
    """
    Fixa todas as fontes de aleatoriedade para reprodutibilidade total
    Cobre: Python random, NumPy, PyTorch (CPU e CUDA) e flags de determinismo do cuDNN
    Args: seed: Valor da semente
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def quantize_input(X: np.ndarray, s_x: float, bits: int) -> np.ndarray:
    """
    Quantiza amostras de entrada: clip(round(x / s_x)).
    Única implementação usada pela emulação em Python, pelo sketch e pelos lotes de teste,
    para que as três vejam exatamente os mesmos inteiros.
    """
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scaled = np.round(np.asarray(X, dtype=np.float64) / s_x)
    return np.clip(scaled, lo, hi).astype(_NP_WEIGHT[bits])


# Configuração centralizada
@dataclass
class PipelineConfig:
    """Parâmetros globais do pipeline. Altere aqui para novos experimentos."""

    # Data
    dataset_path: str = r'C:\Users\jvt\Downloads\data\3W'
    selected_col: str = 'T-TPT'
    target_class: list[int] = field(default_factory=lambda: [3, 4])
    train_ratio: float = 0.8
    # 'well'   : divide por poço físico (nenhum poço aparece em treino e teste ao mesmo
    #            tempo). Eventos simulados/desenhados à mão não têm poço real recorrente
    #            e contam como entidades independentes. Única opção sem vazamento de
    #            entidade -- ver auditoria em .claude/HISTORY.md ("Auditoria de data
    #            leakage").
    # 'event'  : divide por evento/arquivo (janelas do mesmo evento nunca ficam em treino
    #            e teste, mas o MESMO poço pode ter eventos nos dois lados -- o 3W tem só
    #            dezenas de poços para milhares de eventos, então isso praticamente
    #            garante vazamento de poço). Mantido para comparar com 'well' e medir o
    #            quanto a acurácia estava inflada pelo vazamento de entidade.
    # 'window' : sorteia janelas (comportamento antigo; as janelas têm 50% de sobreposição,
    #            então há vazamento entre treino e teste)
    split_mode: str = 'event'

    # Windowing (o rótulo de cada janela é a moda dos rótulos originais nela contidos)
    window_size: int = 1000
    window_type: str = 'hann'
    overlap: float = 0.5
    pad_last_window: bool = True

    # Seed
    random_seed: int = 2026

    # Model
    hidden_sizes: tuple[int, ...] = (32, 16)
    activation_function: str = 'relu'  # chave de _ACTIVATIONS

    # Training
    optimizer: str = 'adam'      # chave de _OPTIMIZERS
    criterion: str = 'cross_entropy'  # chave de _CRITERIA
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001

    # Evaluation
    metrics: list[str] = field(default_factory=lambda: ['accuracy'])
    power_measure_secs: float = 0.1

    # Export
    quantization_bits: list[int] = field(default_factory=lambda: [8, 16, 32])
    n_export: int = 10
    calib_samples: int = 2000
    eval_samples: Optional[int] = None   # None = usa todo o conjunto de teste

    # Modelo de energia do sketch (SUPOSIÇÃO, não medição): P = VCC * I, E = P * tempo
    assumed_vcc: float = 5.0
    assumed_current_a: float = 0.020

    # Output
    output_dir: Path = Path(r'/output')

    # classes
    @property
    def class_codes(self) -> list[int]:
        """Códigos originais do 3W usados (sempre inclui 0 = operação normal)."""
        return sorted({0, *self.target_class})

    @property
    def output_size(self) -> int:
        """Número de saídas da rede = número de classes (rótulos remapeados para 0..K-1)."""
        return len(self.class_codes)

    @property
    def label_map(self) -> dict[int, int]:
        """Código 3W -> índice da saída (ex.: {0: 0, 3: 1, 4: 2})."""
        return {code: i for i, code in enumerate(self.class_codes)}

    @property
    def class_names(self) -> list[str]:
        """Nomes na ordem dos índices de saída."""
        return [CLASS_NAMES_3W.get(c, f'Class {c}') for c in self.class_codes]

    @property
    def assumed_power_w(self) -> float:
        return self.assumed_vcc * self.assumed_current_a

    # diretórios
    @property
    def logs_dir(self) -> Path:
        return self.output_dir / 'logs'

    @property
    def models_dir(self) -> Path:
        return self.output_dir / 'models'

    @property
    def sketches_dir(self) -> Path:
        return self.output_dir / 'sketches'

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / 'plots'


# DataProcessor
@dataclass
class WindowedData:
    """Janelas já rotuladas. ``y`` usa os índices 0..K-1.

    ``groups`` é o id do evento (arquivo) de origem de cada janela. ``wells`` é o id da
    entidade física (poço) de origem: vários eventos reais podem vir do mesmo poço, mas
    eventos simulados/desenhados à mão não têm poço real recorrente e cada um conta como
    sua própria entidade (ver ``DataProcessor._entity_id``).
    """
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    wells: np.ndarray


class DataProcessor:
    """Carrega o dataset 3W, aplica windowing e divide em treino/teste."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _entity_id(file_name, event_type: str) -> str:
        """Id da entidade física de origem de um evento, para agrupar o split por poço.

        Eventos 'real' vêm de um poço físico com vários eventos ao longo do tempo (3W
        2.0.0: 2228 eventos em só 42 poços -- ver auditoria em .claude/HISTORY.md). O
        nome do arquivo é ``<POCO>_<timestamp>.parquet`` (ex.:
        ``WELL-00001_20170201010101.parquet``), então o poço é tudo antes do primeiro
        '_' (mesma convenção usada em ``data_visualization/three_w_chart.py``).

        Eventos 'simulated'/'drawn' não representam um poço real recorrente -- cada
        arquivo é uma instância sintética independente -- então usamos o nome do arquivo
        inteiro como entidade: não agrupa com nenhum outro evento.
        """
        stem = Path(file_name).stem
        if event_type == 'real':
            return stem.split('_', 1)[0]
        return stem

    def load_and_window(self) -> WindowedData:
        """
        Carrega eventos do dataset e aplica janelamento
        Returns: WindowedData com features, rótulos remapeados e id do evento de cada janela
        """
        self._logger.info('Carregando dataset de: %s', self._cfg.dataset_path)
        # `target_class` filtra quais PASTAS de evento são carregadas. Nesta versão do 3W,
        # eventos de pastas 3/4 têm a coluna 'class' só com o código do próprio evento (ou NaN
        # no período de transição) -- nunca com 0. Para ter amostras de Normal Operation é
        # preciso carregar também a pasta 0 explicitamente (`class_codes` = {0, *target_class}).
        ds_config = ParquetDatasetConfig(
            path=self._cfg.dataset_path,
            target_class=self._cfg.class_codes,
            columns=[self._cfg.selected_col],
        )
        ds = ParquetDataset(ds_config)

        # Windowing do ThreeWToolkit janela sinal e rótulo juntos: o rótulo de cada janela
        # é o mais frequente (label_strategy='mode'), que é o equivalente ao antigo
        # "boxcar + moda" usado para as labels.
        windowing = Windowing(
            WindowingConfig(
                window=self._cfg.window_type,
                window_size=self._cfg.window_size,
                overlap=self._cfg.overlap,
                pad_last_window=self._cfg.pad_last_window,
                label_strategy='mode',
            )
        )

        label_map = self._cfg.label_map
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        gs: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        n_dropped_label = 0
        n_dropped_nan = 0

        for ev_id, event in enumerate(ds):
            well_id = self._entity_id(event.metadata['file_name'], event.metadata['event_type'])
            windowed = windowing.transform(event)

            # sinal volta com MultiIndex (window, variable); só há uma variável (selected_col)
            X_ev = (
                windowed.signal.droplevel('variable')
                .sort_index()
                .to_numpy(dtype=np.float32)
            )
            y_raw = windowed.label.to_numpy(dtype=np.float64)
            if len(X_ev) != len(y_raw):
                raise ValueError(
                    f'Evento {ev_id}: {len(X_ev)} janelas de sinal vs {len(y_raw)} de rótulo'
                )

            # remapeia os códigos do 3W (ex.: 0, 3, 4) para índices de saída (0, 1, 2)
            y_idx = np.full(len(y_raw), -1, dtype=np.int64)
            for code, idx in label_map.items():
                y_idx[y_raw == code] = idx
            label_ok = y_idx >= 0

            # ~5.9% das leituras de sensor do 3W real são NaN (sensor ausente/com falha); uma
            # única janela com NaN contamina o forward pass inteiro (e, com Adam, corrompe
            # permanentemente os pesos via NaN no gradiente). Descartamos a janela toda.
            nan_mask = np.isnan(X_ev).any(axis=1)
            keep = label_ok & ~nan_mask
            n_dropped_label += int((~label_ok).sum())
            n_dropped_nan += int((label_ok & nan_mask).sum())

            xs.append(X_ev[keep])
            ys.append(y_idx[keep])
            gs.append(np.full(int(keep.sum()), ev_id, dtype=np.int64))
            ws.append(np.full(int(keep.sum()), well_id, dtype=object))

        if n_dropped_label:
            self._logger.warning(
                '%d janelas descartadas por rótulo fora de %s (ou NaN no rótulo)',
                n_dropped_label, sorted(label_map),
            )
        if n_dropped_nan:
            self._logger.warning(
                '%d janelas descartadas por NaN no sinal (sensor ausente/com falha no 3W)',
                n_dropped_nan,
            )

        data = WindowedData(
            X=np.concatenate(xs, axis=0),
            y=np.concatenate(ys, axis=0),
            groups=np.concatenate(gs, axis=0),
            wells=np.concatenate(ws, axis=0),
        )
        self._logger.info(
            'Windowing concluido: %d janelas de %d eventos (%d poços/entidades) | classes %s',
            len(data.y), len(np.unique(data.groups)), len(np.unique(data.wells)),
            {self._cfg.class_names[int(c)]: int((data.y == c).sum()) for c in np.unique(data.y)},
        )
        return data

    def split(self, data: WindowedData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide em treino e teste
        Args:    data: WindowedData
        Returns: Tupla (X_train, y_train, X_test, y_test)
        """
        n_total = len(data.y)
        rng = np.random.default_rng(self._cfg.random_seed)

        if self._cfg.split_mode == 'window':
            self._logger.warning(
                "split_mode='window': janelas vizinhas (50%% de sobreposição) caem em treino e "
                "teste, o que infla a acurácia. Prefira split_mode='well'."
            )
            idx = rng.permutation(n_total)
            n_train = int(self._cfg.train_ratio * n_total)
            train_idx, test_idx = idx[:n_train], idx[n_train:]
        elif self._cfg.split_mode == 'event':
            self._logger.warning(
                "split_mode='event': o mesmo poço pode ter eventos em treino e teste "
                "(o 3W tem só dezenas de poços para milhares de eventos), o que infla a "
                "acurácia. Prefira split_mode='well'."
            )
            train_idx, test_idx = self._split_by_key(data.groups, n_total, rng)
        elif self._cfg.split_mode == 'well':
            train_idx, test_idx = self._split_by_key(data.wells, n_total, rng)
        else:
            raise ValueError(
                f"split_mode inválido: {self._cfg.split_mode!r} (use 'well', 'event' ou 'window')"
            )

        X_train, y_train = data.X[train_idx], data.y[train_idx]
        X_test,  y_test  = data.X[test_idx],  data.y[test_idx]

        self._logger.info(
            'Split (%s): %d treino / %d teste | treino %s | teste %s',
            self._cfg.split_mode, len(train_idx), len(test_idx),
            {int(c): int((y_train == c).sum()) for c in np.unique(y_train)},
            {int(c): int((y_test == c).sum()) for c in np.unique(y_test)},
        )
        missing = set(range(self._cfg.output_size)) - set(np.unique(y_test).tolist())
        if missing:
            self._logger.warning('Classes ausentes no teste: %s', sorted(missing))

        return X_train, y_train, X_test, y_test

    def _split_by_key(
        self, key: np.ndarray, n_total: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Divide índices de janela em treino/teste agrupando por ``key`` (evento ou poço).

        Sorteia a ordem dos grupos e acumula grupos inteiros no treino até atingir
        ``train_ratio``; nenhum grupo fica partido entre os dois conjuntos. Usada tanto
        por ``split_mode='event'`` (``key=data.groups``) quanto por
        ``split_mode='well'`` (``key=data.wells``) -- a única diferença entre os dois
        modos é qual coluna de ``WindowedData`` define a unidade de independência.
        """
        groups = np.unique(key)
        rng.shuffle(groups)
        counts = {g: int((key == g).sum()) for g in groups}
        target = self._cfg.train_ratio * n_total
        train_groups, acc = [], 0
        for g in groups:
            if acc >= target:
                break
            train_groups.append(g)
            acc += counts[g]
        is_train = np.isin(key, train_groups)
        return np.where(is_train)[0], np.where(~is_train)[0]

    def normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Padroniza (z-score, escalar único) usando média/desvio do treino.

        O sinal bruto do 3W (ex.: T-TPT) fica em torno de dezenas/centenas e nunca é
        normalizado antes da MLP (pesos LazyLinear inicializados para entrada ~N(0,1)).
        Sem isso, em treinos longos com o dataset completo (centenas de milhares de passos
        de otimização) os logits divergem e o loss vira NaN. Usamos um único (mu, sigma)
        escalar — não por coluna — porque o Hann windowing deixa as colunas nas bordas de
        cada janela perto de zero; normalizar por coluna faria o desvio ali tender a 0 e
        explodiria o ruído nessas posições.
        """
        mu = float(X_train.mean())
        sigma = float(X_train.std()) + 1e-8
        self._logger.info('Normalizando entrada (z-score): mu=%.4f sigma=%.4f (estatísticas do treino)', mu, sigma)
        return (X_train - mu) / sigma, (X_test - mu) / sigma


# Adaptador: expõe janelas já prontas (arrays NumPy) como ``BaseDataset`` do ThreeWToolkit.
# O ``TorchTrainer`` só aceita ``BaseDataset``, não arrays/DataFrames crus. Usamos um único
# "evento" cobrindo todas as janelas, já que ``_prepare_data`` concatena os eventos do
# dataset ao longo do eixo 0.
class ArrayDataset(BaseDataset):
    """Adapta um par ``(X, y)`` de janelas já extraídas para a interface ``BaseDataset``."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self._signal = pd.DataFrame(np.asarray(X, dtype=np.float32))
        self._label = pd.Series(np.asarray(y, dtype=np.int64), name='label')

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> DatasetOutputs:
        if idx != 0:
            raise IndexError(idx)
        return DatasetOutputs(signal=self._signal, label=self._label)


# ModelManager
class ModelManager:
    """Responsável pelo treinamento e avaliação do modelo MLP."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.trainer: Optional[TorchTrainer] = None
        self.training_result: Optional[TrainingResult] = None

    def build_trainer(self) -> TorchTrainer:
        """Instancia o ``TorchTrainer`` com os hiperparâmetros da config."""
        mlp_config = MLPConfig(
            hidden_sizes=self._cfg.hidden_sizes,
            output_size=self._cfg.output_size,
            activation_function=_ACTIVATIONS[self._cfg.activation_function],
        )
        trainer_config = TorchTrainerConfig(
            config_model=mlp_config,
            batch_size=self._cfg.batch_size,
            epochs=self._cfg.epochs,
            optimizer=_OPTIMIZERS[self._cfg.optimizer],
            learning_rate=self._cfg.learning_rate,
            criterion=_CRITERIA[self._cfg.criterion],
            seed=self._cfg.random_seed,
            device='cpu',
        )
        self.trainer = TorchTrainer(trainer_config)
        return self.trainer

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Treina o modelo
        Args: X_train: Features de treino | y_train: Rótulos de treino
        """
        if self.trainer is None:
            self.build_trainer()

        self._logger.info('Iniciando treinamento (%d epocas).', self._cfg.epochs)
        self.training_result = self.trainer.train(ArrayDataset(X_train, y_train))
        self._logger.info('Treinamento concluido.')

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Avalia o modelo com rastreamento de emissões de CO_2
        Atenção: tempo e energia aqui são do COMPUTADOR (avaliação completa, incluindo
        overhead de DataFrame/metrics), não do ESP32. Não compare direto com o sketch.
        Args:    X_test: Features de teste | y_test: Rótulos de teste
        Returns: Dicionário com métricas, tempos e consumo energético
        """
        test_dataset = ArrayDataset(X_test, y_test)
        assessment_config = ModelAssessmentConfig(
            metrics=self._cfg.metrics, output_dir=self._cfg.output_dir, export_results=False,
        )
        tracker = EmissionsTracker(measure_power_secs=self._cfg.power_measure_secs, save_to_file=True, output_dir=self._cfg.output_dir)

        t0 = time.time()
        tracker.start()
        predictions = self.trainer.predict(test_dataset)
        tracker.stop()
        duration = time.time() - t0

        assess_out = ModelAssessment(assessment_config).evaluate(self.training_result, predictions)

        energy_kwh    = tracker.final_emissions_data.energy_consumed
        energy_wh     = energy_kwh * 1000
        energy_joules = energy_wh * 3600
        n_samples     = len(X_test)
        emissions_kg  = tracker.final_emissions_data.emissions

        report = {
            "metrics":          assess_out.metrics,
            "duration_s":       duration,
            "n_samples":        n_samples,
            "energy_joules":    energy_joules,
            "energy_wh":        energy_wh,
            "co2_kg":           emissions_kg,
            "us_per_sample":    duration / n_samples * 1e6,
            "uj_per_sample":    energy_joules / n_samples * 1e6,
            "true_values":      predictions.y_true,
            "predictions":      predictions.y_pred,
        }

        self._logger.info('Métricas de teste        : %s', report["metrics"])
        self._logger.info('Tempo total de inferência (FP32, PC)  : %.3f s', report["duration_s"])
        self._logger.info('Tempo médio por amostra (FP32, PC)    : %.2f µs', report["us_per_sample"])
        self._logger.info('Energia consumida (FP32, PC)          : %.4f J  /  %.6f Wh', report["energy_joules"], report["energy_wh"])
        self._logger.info('Energia média por amostra (FP32, PC)  : %.4f µJ', report["uj_per_sample"])
        self._logger.info('CO2 estimado                          : %.6f kg', report["co2_kg"])
        return report

    @property
    def model(self) -> MLP:
        return self.trainer.model


# Quantizer
class Quantizer:
    """
    Quantização pós-treinamento (PTQ) para INT8, INT16 e INT32.

    Toda a parte numérica é NumPy puro e espelha, operação a operação, o que o sketch faz
    no ESP32 (arredondamento ao mais próximo na requantização, ReLU depois do clip e
    argmax sobre o acumulador bruto da última camada).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)

    # helpers estáticos

    @staticmethod
    def _get_all_linear_layers(mlp: MLP) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extrai pesos e biases das camadas ``Linear`` do MLP (float64, W em [in, out]).

        Valida que a rede é Linear -> ReLU -> ... -> Linear, que é o que o sketch implementa.
        """
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        kinds: list[str] = []
        for module in mlp.model:
            kind = type(module).__name__
            if kind == 'Linear':
                W = module.weight.detach().cpu().numpy().T.astype(np.float64)  # [in, out]
                b = module.bias.detach().cpu().numpy().astype(np.float64)      # [out]
                layers.append((W, b))
                kinds.append('Linear')
            elif kind == 'ReLU':
                kinds.append('ReLU')
            elif kind in ('Dropout', 'Identity', 'Flatten'):
                continue  # sem efeito em inferência
            else:
                raise NotImplementedError(
                    f'Camada {kind} não é suportada pelo quantizador/sketch (só Linear e ReLU)'
                )
        expected = ['Linear', 'ReLU'] * (len(layers) - 1) + ['Linear']
        if kinds != expected:
            raise NotImplementedError(f'Arquitetura esperada Linear/ReLU alternados, obtida {kinds}')
        return layers

    @staticmethod
    def forward_float(
        layers: list[tuple[np.ndarray, np.ndarray]], X: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Inferência em ponto flutuante (batch), capturando as ativações ocultas pós-ReLU.

        Returns: Tupla ``(hidden_activations, logits)``.
        """
        h = np.asarray(X, dtype=np.float64)
        hidden: list[np.ndarray] = []
        logits = h
        for l, (W, b) in enumerate(layers):
            z = h @ W + b
            if l < len(layers) - 1:
                h = np.maximum(z, 0.0)
                hidden.append(h)
            else:
                logits = z
        return hidden, logits

    @staticmethod
    def _calc_scale(arr: np.ndarray, bits: int) -> float:
        """Escala de quantização baseada no valor absoluto máximo."""
        return float(np.max(np.abs(arr))) / (2 ** (bits - 1) - 1) + _EPS

    @staticmethod
    def _cast_checked(arr: np.ndarray, dtype: type, name: str, logger: logging.Logger) -> np.ndarray:
        """Converte para inteiro avisando (e saturando) se o valor não cabe no tipo."""
        info = np.iinfo(dtype)
        lim_hi = float(info.max) if info.bits <= 32 else 2.0 ** 62
        lim_lo = float(info.min) if info.bits <= 32 else -(2.0 ** 62)
        if arr.size and (arr.max() > lim_hi or arr.min() < lim_lo):
            logger.warning('%s não cabe em %s: saturando', name, np.dtype(dtype).name)
            arr = np.clip(arr, lim_lo, lim_hi)
        return arr.astype(dtype)

    @staticmethod
    def quantize_multiplier(m: float) -> tuple[int, int]:
        """Converte um multiplicador real em (mult, shift) com mult em Q31.

        Vale: m ≈ mult / 2^31 / 2^shift. ``shift`` negativo significa deslocamento à esquerda
        (multiplicadores >= 1).
        """
        if m <= 0.0:
            return 0, 0
        shift = 0
        while m >= 1.0:
            m /= 2.0
            shift -= 1
        while m < 0.5 and shift < 31:
            m *= 2.0
            shift += 1
        mult = int(np.round(m * (1 << 31)))
        if mult >= (1 << 31):      # m arredondou para 1.0
            mult >>= 1
            shift -= 1
        return mult, shift

    @staticmethod
    def fp_requant(acc: int, mult: int, shift: int, out_min: int, out_max: int) -> int:
        """Re-quantização de ponto fixo com arredondamento ao mais próximo e clipping.

        Versão escalar com inteiros exatos de Python (referência para testes).
        """
        x = (int(acc) * mult + (1 << 30)) >> 31
        if shift > 0:
            x = (x + (1 << (shift - 1))) >> shift
        elif shift < 0:
            x = x * (1 << (-shift))
        return max(out_min, min(out_max, x))

    @staticmethod
    def requant_np(acc: np.ndarray, mult: int, shift: int, out_min: int, out_max: int) -> np.ndarray:
        """Versão vetorizada de ``fp_requant``, idêntica ao ``mulq31``/``requant`` do sketch.

        Divide o acumulador em parte alta/baixa para o produto caber em int64.
        """
        a = np.asarray(acc, dtype=np.int64)
        a_hi = a >> 31
        a_lo = a & 0x7FFFFFFF
        m = np.int64(mult)
        x = a_hi * m + ((a_lo * m + (1 << 30)) >> 31)
        if shift > 0:
            x = (x + (1 << (shift - 1))) >> shift
        elif shift < 0:
            x = x * (1 << (-shift))
        return np.clip(x, out_min, out_max)

    @staticmethod
    def dequantize(q_int: np.ndarray, scale: float) -> np.ndarray:
        """Dequantiza um array inteiro para float64."""
        return q_int.astype(np.float64) * scale

    @staticmethod
    def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
        cm = np.zeros((n, n), dtype=np.int64)
        np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
        return cm

    # inferência inteira (espelho do sketch)

    @classmethod
    def int_forward(cls, q: dict, X: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, list[int]]:
        """Inferência 100% inteira, igual à do ESP32.

        Returns: (acumuladores brutos da última camada [N, saídas], |acc| máximo por camada)
        """
        bits = q['bits']
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        n_layers = q['n_layers']
        X = np.asarray(X)
        outs: list[np.ndarray] = []
        max_abs = [0] * n_layers
        for s in range(0, len(X), chunk):
            h = quantize_input(X[s:s + chunk], q['s_x'], bits).astype(np.int64)
            for l in range(n_layers):
                z = h @ q['W_q'][l].astype(np.int64) + q['b_q'][l].astype(np.int64)
                max_abs[l] = max(max_abs[l], int(np.max(np.abs(z))))
                if l < n_layers - 1:
                    h = np.maximum(cls.requant_np(z, q['MULTS'][l], q['SHIFTS'][l], lo, hi), 0)
                else:
                    outs.append(z)
        return np.concatenate(outs, axis=0), max_abs

    @classmethod
    def int_predict(cls, q: dict, X: np.ndarray) -> np.ndarray:
        """Classe prevista pela inferência inteira (argmax do acumulador, como no sketch)."""
        logits, _ = cls.int_forward(q, X)
        return np.argmax(logits, axis=1)

    # método principal

    def quantize_model(self, mlp: MLP, X_calib: np.ndarray, bits: int) -> dict:
        """Realiza a quantização completa do modelo para ``bits`` bits.

        Utiliza um subconjunto de calibração (sorteado com um gerador local, então
        independe do estado global do NumPy) para estimar as escalas das ativações.

        Args:
            mlp:     Modelo MLP treinado.
            X_calib: Conjunto de calibração (features de treino).
            bits:    Largura de bits alvo (8, 16 ou 32).

        Returns:
            Dicionário com pesos quantizados, escalas e multiplicadores.
        """
        self._logger.info("Quantizando modelo - INT%d", bits)

        layers = self._get_all_linear_layers(mlp)
        n_layers = len(layers)
        clip_max = 2 ** (bits - 1) - 1
        clip_min = -(2 ** (bits - 1))
        s_W = [self._calc_scale(W, bits) for W, _ in layers]

        rng = np.random.default_rng(self._cfg.random_seed)
        n_cal = min(self._cfg.calib_samples, len(X_calib))
        calib_idx = rng.choice(len(X_calib), size=n_cal, replace=False)
        Xc = np.asarray(X_calib[calib_idx], dtype=np.float64)

        hidden, _ = self.forward_float(layers, Xc)
        s_x = float(np.max(np.abs(Xc))) / clip_max + _EPS
        s_h = [self._calc_scale(h, bits) for h in hidden]
        s_in = [s_x] + s_h

        W_q, b_q, MULTS, SHIFTS = [], [], [], []
        for l, (W, b) in enumerate(layers):
            W_q.append(np.clip(np.round(W / s_W[l]), clip_min, clip_max).astype(_NP_WEIGHT[bits]))
            b_q.append(self._cast_checked(
                np.round(b / (s_in[l] * s_W[l])), _NP_BIAS[bits], f'b{l + 1}', self._logger
            ))
            if l < n_layers - 1:   # a última camada fica no acumulador bruto (sem requant)
                mult, shift = self.quantize_multiplier((s_in[l] * s_W[l]) / s_h[l])
                MULTS.append(mult)
                SHIFTS.append(shift)

        # limite teórico do acumulador: soma(|W|)*max|x| + |b|, por neurônio de saída
        acc_limit = 2.0 ** _ACC_BITS[bits] - 1
        overflow_warnings: list[str] = []
        for l in range(n_layers):
            bound = float(np.max(
                np.abs(W_q[l].astype(np.float64)).sum(axis=0) * (2 ** (bits - 1))
                + np.abs(b_q[l].astype(np.float64))
            ))
            if bound > acc_limit:
                msg = (f'camada {l + 1}: pior caso do acumulador 2^{np.log2(bound):.1f} '
                       f'> limite 2^{_ACC_BITS[bits]} (sem overflow só se os dados reais ficarem abaixo)')
                overflow_warnings.append(msg)
                self._logger.warning('INT%d %s', bits, msg)

        flash_kb = sum(W.nbytes + b.nbytes for W, b in zip(W_q, b_q)) / 1024
        self._logger.info("INT%d - Parâmetros: ~%.1f KB | calibração: %d amostras", bits, flash_kb, n_cal)

        return dict(
            bits=bits,
            n_layers=n_layers,
            input_size=layers[0][0].shape[0],
            hidden_sizes=[W.shape[1] for W, _ in layers[:-1]],
            output_size=layers[-1][0].shape[1],
            W_q=W_q, b_q=b_q,
            s_x=s_x, s_h=s_h, s_W=s_W, s_acc_out=s_in[-1] * s_W[-1],
            MULTS=MULTS, SHIFTS=SHIFTS,
            flash_kb=flash_kb,
            overflow_warnings=overflow_warnings,
        )

    def evaluate_quantized(
        self,
        q: dict,
        mlp: MLP,
        X_eval: np.ndarray,
        y_eval: Optional[np.ndarray] = None,
        n_max: Optional[int] = None,
    ) -> dict:
        """Avalia o modelo quantizado com a inferência inteira (espelho do ESP32).

        Compara com o FP32 nas MESMAS amostras: acurácia, divergência de predições,
        SNR/RMSE dos logits (acumulador dequantizado) e matriz de confusão.
        """
        bits = q['bits']
        layers = self._get_all_linear_layers(mlp)
        if n_max:
            X_eval = X_eval[:n_max]
            y_eval = None if y_eval is None else y_eval[:n_max]

        fp_logits = np.concatenate(
            [self.forward_float(layers, X_eval[s:s + 4096])[1] for s in range(0, len(X_eval), 4096)], axis=0
        )
        q_logits, max_abs = self.int_forward(q, X_eval)
        fp_pred = np.argmax(fp_logits, axis=1)
        q_pred = np.argmax(q_logits, axis=1)

        deq = q_logits.astype(np.float64) * q['s_acc_out']
        noise = fp_logits - deq
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = 10.0 * np.log10(np.mean(fp_logits ** 2, axis=1) / np.mean(noise ** 2, axis=1))
        finite = np.isfinite(snr)

        acc_bits = [float(np.log2(m)) if m > 0 else 0.0 for m in max_abs]
        out = dict(
            bits=bits,
            n_samples=int(len(X_eval)),
            divergence_pct=float(np.mean(fp_pred != q_pred) * 100),
            snr_db=float(np.mean(snr[finite])) if finite.any() else float('inf'),
            rmse=float(np.sqrt(np.mean(noise ** 2))),
            max_abs_acc_log2=acc_bits,
            acc_limit_log2=_ACC_BITS[bits],
            flash_kb=q['flash_kb'],
        )
        if y_eval is not None:
            n = q['output_size']
            cm_q = self._confusion(y_eval, q_pred, n)
            cm_fp = self._confusion(y_eval, fp_pred, n)
            norm = lambda cm: cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            out.update(
                acc_int_pct=float(np.mean(q_pred == y_eval) * 100),
                acc_fp32_pct=float(np.mean(fp_pred == y_eval) * 100),
                confusion_counts=cm_q.tolist(),
                confusion_norm=norm(cm_q).tolist(),
                confusion_norm_fp32=norm(cm_fp).tolist(),
            )

        self._logger.info(
            "INT%d (%d amostras) - Acurácia: %s | FP32 nas mesmas amostras: %s | Divergência: %.1f%% | "
            "SNR: %.2f dB | RMSE: %.6f | |acc| máx: 2^%.1f (limite 2^%d)",
            bits, len(X_eval),
            f"{out['acc_int_pct']:.2f}%" if y_eval is not None else 'n/d',
            f"{out['acc_fp32_pct']:.2f}%" if y_eval is not None else 'n/d',
            out['divergence_pct'], out['snr_db'], out['rmse'], max(acc_bits), _ACC_BITS[bits],
        )
        return out

    @staticmethod
    def save(q: dict, path: Path) -> Path:
        """Salva pesos, escalas e multiplicadores (rastreabilidade: o que foi para o sketch)."""
        arrays = {f'W{l + 1}_q': W for l, W in enumerate(q['W_q'])}
        arrays.update({f'b{l + 1}_q': b for l, b in enumerate(q['b_q'])})
        np.savez(path, bits=q['bits'], s_x=q['s_x'], s_h=np.array(q['s_h']), s_W=np.array(q['s_W']),
                 s_acc_out=q['s_acc_out'], MULTS=np.array(q['MULTS']), SHIFTS=np.array(q['SHIFTS']),
                 **arrays)
        return Path(path)


# ArduinoExporter
class ArduinoExporter:
    """Gera sketches Arduino (.ino) a partir de modelos quantizados."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)

    # ── helpers internos ────────────────────────────────────────────────

    @staticmethod
    def _c_array(name: str, arr: np.ndarray, c_type: str, per_line: int = 16) -> str:
        """Serializa um array NumPy como array C (flash mapeada em memória no ESP32 -- sem
        PROGMEM/pgm_read_*, que são específicos de AVR e nem sempre existem no núcleo ESP32)."""
        flat = [int(v) for v in arr.flatten()]
        rows = [
            "  " + ", ".join(map(str, flat[i : i + per_line]))
            for i in range(0, len(flat), per_line)
        ]
        return f"const {c_type} {name}[] = {{\n" + ",\n".join(rows) + "\n};\n\n"

    @staticmethod
    def _read_weight(ca: str, arr_name: str, flat_idx: str) -> str:
        """Leitura de peso: indexação direta (o array já é do tipo com sinal correto)."""
        return f"(({ca}){arr_name}[{flat_idx}])"

    @staticmethod
    def _read_bias(ca: str, arr_name: str, idx: str) -> str:
        """Leitura de bias: indexação direta."""
        return f"(({ca}){arr_name}[{idx}])"

    # ── construção das seções do sketch ─────────────────────────────────

    def _build_defines(self, q: dict, N: int) -> str:
        """Gera a seção de ``#define`` do sketch."""
        defines = (
            "#include <Arduino.h>\n\n"
            f"#define INPUT_SIZE  {q['input_size']}\n"
        )
        for l, h in enumerate(q["hidden_sizes"]):
            defines += f"#define HIDDEN{l + 1}     {h}\n"
        defines += f"#define OUTPUT_SIZE {q['output_size']}\n"
        defines += f"#define N_TEST      {N}\n\n"
        for l in range(q["n_layers"] - 1):   # a última camada não tem requantização
            defines += f"#define MULT{l + 1}  ((int32_t){q['MULTS'][l]}L)\n"
            defines += f"#define SHIFT{l + 1} {q['SHIFTS'][l]}\n"
        defines += "\n"
        return defines

    def _build_data_section(
        self, q: dict, X_te_q: np.ndarray, y_te: np.ndarray, N: int,
        expected_pred: Optional[np.ndarray],
    ) -> str:
        """Gera os arrays (const, em flash) de pesos, biases e amostras de teste."""
        bits = q["bits"]
        cw = _C_WEIGHT[bits]
        cb = _C_BIAS[bits]
        data_sec = ""
        for l in range(q["n_layers"]):
            data_sec += self._c_array(f"W{l + 1}_q", q["W_q"][l], cw)
            data_sec += self._c_array(f"b{l + 1}_q", q["b_q"][l], cb)
        data_sec += f"// {N} amostras de teste\n"
        data_sec += self._c_array("X_test_q", X_te_q, cw)
        data_sec += "// rótulos (índices 0..K-1): " + ", ".join(
            f"{i}={n}" for i, n in enumerate(self._cfg.class_names)) + "\n"
        data_sec += self._c_array("y_labels", y_te[:N].astype(np.uint8), "uint8_t")
        if expected_pred is not None:
            data_sec += "// predição esperada (emulação inteira em Python, bit a bit)\n"
            data_sec += self._c_array("y_expected", np.asarray(expected_pred)[:N].astype(np.uint8), "uint8_t")
        return data_sec

    def _build_requant_block(self, bits: int) -> str:
        """Gera a função ``requant`` em C (Q31 com arredondamento ao mais próximo)."""
        if bits == 8:
            return (
                "static int8_t requant(int32_t acc, int32_t mult, int shift) {\n"
                "  int64_t x = ((int64_t)acc * mult + (1LL << 30)) >> 31;\n"
                "  if (shift > 0)      x = (x + (1LL << (shift - 1))) >> shift;\n"
                "  else if (shift < 0) x = x * (1LL << (-shift));\n"
                "  if (x >  127) x =  127;\n"
                "  if (x < -128) x = -128;\n"
                "  return (int8_t)x;\n"
                "}\n\n"
            )
        out_type = "int16_t" if bits == 16 else "int32_t"
        hi_val   = "32767"   if bits == 16 else "2147483647LL"
        lo_val   = "-32768"  if bits == 16 else "-2147483648LL"
        return (
            f"// round((a * m) / 2^31) sem estourar int64: a = a_hi * 2^31 + a_lo\n"
            f"static int64_t mulq31(int64_t a, int32_t m) {{\n"
            f"  int64_t a_hi = a >> 31;\n"
            f"  int64_t a_lo = a & 0x7FFFFFFFLL;\n"
            f"  return a_hi * m + ((a_lo * m + (1LL << 30)) >> 31);\n"
            f"}}\n\n"
            f"static {out_type} requant(int64_t acc, int32_t mult, int shift) {{\n"
            f"  int64_t x = mulq31(acc, mult);\n"
            f"  if (shift > 0)      x = (x + (1LL << (shift - 1))) >> shift;\n"
            f"  else if (shift < 0) x = x * (1LL << (-shift));\n"
            f"  if (x >  {hi_val}) x =  {hi_val};\n"
            f"  if (x < {lo_val}) x = {lo_val};\n"
            f"  return ({out_type})x;\n"
            f"}}\n\n"
        )

    def _build_infer_body(self, q: dict) -> str:
        """Gera a função ``mlp_predict`` com laços desenrolados por camada."""
        bits = q["bits"]
        n_layers = q["n_layers"]
        hidden_sizes = q["hidden_sizes"]
        input_size = q["input_size"]
        output_size = q["output_size"]
        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        cw = _C_WEIGHT[bits]
        ca = _C_ACCUM[bits]

        body = f"int mlp_predict(const {cw} *x_in) {{\n\n"
        prev_buf = "x_in"

        for l in range(n_layers):
            is_output = l == n_layers - 1
            W_name    = f"W{l + 1}_q"
            b_name    = f"b{l + 1}_q"
            mult_name = f"MULT{l + 1}"
            shift_name = f"SHIFT{l + 1}"

            if is_output:
                buf_name   = "logits"
                buf_decl   = f"  {ca} {buf_name}[OUTPUT_SIZE];\n"
                size_macro = "OUTPUT_SIZE"
            else:
                buf_name   = f"z{l + 1}"
                h_macro    = f"HIDDEN{l + 1}"
                buf_decl   = f"  {ca} {buf_name}[{h_macro}];\n"
                size_macro = h_macro

            prev_layer_macro = "INPUT_SIZE" if l == 0 else f"HIDDEN{l}"
            comment = (
                f"  // Layer {l + 1}: z = W{l + 1}^T * "
                + ("x_in" if l == 0 else prev_buf)
                + f" + b{l + 1}"
            )
            if not is_output:
                comment += f", {buf_name} = ReLU(requant(z))"
            else:
                comment += "  (raw accumulator - argmax invariant to scale)"

            rw = self._read_weight(ca, W_name, f"i * {size_macro} + j")
            rb = self._read_bias(ca, b_name, "j")
            rx = f"({ca})x_in[i]" if l == 0 else f"({ca}){prev_buf}[i]"

            body += comment + "\n"
            body += buf_decl
            body += f"  for (int j = 0; j < {size_macro}; j++) {{\n"
            body += f"    {ca} acc = {rb};\n"
            body += f"    for (int i = 0; i < {prev_layer_macro}; i++)\n"
            body += f"      acc += {rx} * {rw};\n"
            if not is_output:
                body += f"    {cw} v = requant(acc, {mult_name}, {shift_name});\n"
                body += f"    {buf_name}[j] = (v > 0) ? ({ca})v : 0;  // ReLU\n"
            else:
                body += f"    {buf_name}[j] = acc;  // mantém acumulador para argmax\n"
            body += "  }\n\n"
            prev_buf = buf_name

        body += (
            "  // Argmax\n"
            "  int best = 0;\n"
            "  for (int k = 1; k < OUTPUT_SIZE; k++)\n"
            "    if (logits[k] > logits[best]) best = k;\n"
            "  return best;\n"
            "}\n\n"
        )
        return body

    def _build_setup_loop(self, q: dict, has_expected: bool) -> str:
        """Gera as funções ``setup()`` e ``loop()`` do sketch."""
        bits = q["bits"]
        cw   = _C_WEIGHT[bits]
        rx_test = f"X_test_q[n * INPUT_SIZE + i]"
        check_decl = "  int mismatch = 0;\n" if has_expected else ""
        check_loop = (
            "    int expected = (int)y_expected[n];\n"
            "    if (pred != expected) mismatch++;\n"
        ) if has_expected else ""
        check_end = (
            '  Serial.print("Divergencia vs emulacao Python: "); Serial.print(mismatch); Serial.print("/"); Serial.println(N_TEST);\n'
        ) if has_expected else ""
        return (
            "// ENERGIA ESTIMADA, NAO MEDIDA: o simulador nao mede consumo.\n"
            "// Assume potencia constante P = VCC * I_ACTIVE e E = P * tempo.\n"
            f"const float VCC      = {self._cfg.assumed_vcc};\n"
            f"const float I_ACTIVE = {self._cfg.assumed_current_a};\n"
            "const float POWER_W  = VCC * I_ACTIVE;\n\n"
            f"void setup() {{\n"
            "  Serial.begin(115200);\n"
            f'  Serial.println("Validacao MLP INT{bits}");\n\n'
            "  int correct = 0;\n"
            f"{check_decl}"
            "  float total_joules  = 0;\n"
            "  unsigned long total_time_us = 0;\n\n"
            "  for (int n = 0; n < N_TEST; n++) {\n"
            f"    {cw} sample[INPUT_SIZE];\n"
            "    for (int i = 0; i < INPUT_SIZE; i++)\n"
            f"      sample[i] = {rx_test};\n\n"
            "    unsigned long t_start = micros();\n"
            "    int pred = mlp_predict(sample);\n"
            "    unsigned long t_end   = micros();\n\n"
            "    unsigned long duration_us = t_end - t_start;\n"
            "    total_time_us += duration_us;\n"
            "    float duration_sec  = duration_us / 1000000.0;\n"
            "    float energy_joules = POWER_W * duration_sec;\n"
            "    total_joules += energy_joules;\n\n"
            "    int label = (int)y_labels[n];\n"
            "    if (pred == label) correct++;\n"
            f"{check_loop}\n"
            '    Serial.print("Amostra "); Serial.print(n);\n'
            '    Serial.print(" | Pred: ");    Serial.print(pred);\n'
            '    Serial.print(" | Real: ");    Serial.print(label);\n'
            '    Serial.print(" | Tempo: ");   Serial.print(duration_us);   Serial.print("us");\n'
            '    Serial.print(" | Energia (estimada): "); Serial.print(energy_joules, 10); Serial.println(" J");\n'
            "  }\n\n"
            "  float avg_time_us = (float)total_time_us / N_TEST;\n"
            "  float avg_joules  = total_joules / N_TEST;\n\n"
            '  Serial.println("------------------------------------------");\n'
            '  Serial.print("Acuracia Final: "); Serial.print(correct); Serial.print("/"); Serial.println(N_TEST);\n'
            f"{check_end}"
            '  Serial.print("Tempo Medio por Amostra: "); Serial.print(avg_time_us); Serial.println(" us");\n'
            '  Serial.print("Energia Media por Amostra (estimada): "); Serial.print(avg_joules, 10); Serial.println(" J");\n'
            '  Serial.print("Tempo Total: "); Serial.print(total_time_us / 1000.0); Serial.println(" ms");\n'
            '  Serial.print("Energia Total (estimada): "); Serial.print(total_joules, 8); Serial.println(" J");\n'
            '  Serial.println("------------------------------------------");\n'
            "}\n\n"
            "void loop() {}\n"
        )

    # ── método público ───────────────────────────────────────────────────

    def export_sketch(
        self,
        q: dict,
        X_te: np.ndarray,
        y_te: np.ndarray,
        *,
        expected_pred: Optional[np.ndarray] = None,
        suffix: str = "",
        n: Optional[int] = None,
    ) -> Path:
        """Gera e salva o sketch Arduino para um modelo quantizado.

        Args:
            q:             Dicionário de quantização retornado por ``Quantizer.quantize_model``.
            X_te:          Amostras de teste (float) para embutir no sketch.
            y_te:          Rótulos correspondentes (índices 0..K-1).
            expected_pred: Predições da emulação inteira; se dadas, o sketch confere uma a uma
                           e imprime a divergência contra a emulação.
            suffix:        Sufixo do nome do arquivo (ex.: ``_batch03``).
            n:             Número de amostras (padrão: ``n_export`` da config).

        Returns:
            Caminho do arquivo ``.ino`` gerado.
        """
        bits = q["bits"]
        N = min(n or self._cfg.n_export, len(X_te))

        X_te_q = quantize_input(X_te[:N], q["s_x"], bits)

        sketch = "".join([
            self._build_defines(q, N),
            self._build_data_section(q, X_te_q, y_te, N, expected_pred),
            self._build_requant_block(bits),
            self._build_infer_body(q),
            self._build_setup_loop(q, expected_pred is not None),
        ])

        self._cfg.sketches_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._cfg.sketches_dir / f"mlp_3w_int{bits}{suffix}.ino"
        out_path.write_text(sketch, encoding="utf-8")

        self._logger.info(
            "Sketch gerado: %s  |  Parâmetros ~%.1f KB  |  %d amostras de teste",
            out_path, q["flash_kb"], N,
        )
        return out_path

    @staticmethod
    def select_balanced_export(
        X: np.ndarray, y: np.ndarray, n: int, seed: int = 2026
    ) -> tuple[np.ndarray, np.ndarray]:
        """Seleciona ``n`` amostras balanceadas entre as classes.

        Garante ≈ ``n // n_classes`` amostras por classe; se uma classe tiver menos que a
        sua cota, as vagas que sobram são distribuídas entre as demais classes.

        Args:
            X:    Features.
            y:    Rótulos.
            n:    Número total de amostras desejadas.
            seed: Semente para reprodutibilidade.

        Returns:
            Tupla ``(X_out, y_out)`` com exatamente ``min(n, len(X))`` linhas.
        """
        rng     = np.random.default_rng(seed)
        classes = np.unique(y)
        n_cls   = len(classes)
        n       = min(n, len(X))
        base, extra = divmod(n, n_cls)

        avail = {c: np.where(y == c)[0] for c in classes}
        take  = {c: min(base + (1 if i < extra else 0), len(avail[c])) for i, c in enumerate(classes)}
        left  = n - sum(take.values())
        while left > 0:
            progressed = False
            for c in classes:
                if left > 0 and take[c] < len(avail[c]):
                    take[c] += 1
                    left -= 1
                    progressed = True
            if not progressed:
                break

        chosen: list[int] = []
        for c in classes:
            chosen.extend(rng.choice(avail[c], size=take[c], replace=False).tolist())

        chosen_arr = np.array(chosen)
        rng.shuffle(chosen_arr)
        return X[chosen_arr], y[chosen_arr]


# Images
class Plotter:
    """Gera e salva visualizações de treinamento e avaliação do modelo."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._apply_scientific_style()

    @staticmethod
    def _apply_scientific_style() -> None:
        """Aplica estilo de artigo científico aos plots matplotlib."""
        plt.rcParams.update({
            'font.size':        12,
            'font.family':      'serif',
            'axes.titlesize':   13,
            'axes.labelsize':   12,
            'xtick.labelsize':  11,
            'ytick.labelsize':  11,
            'legend.fontsize':  11,
            'figure.dpi':       150,
            'axes.grid':        True,
            'grid.linestyle':   '--',
            'grid.alpha':       0.4,
            'lines.linewidth':  1.5,
        })

    def plot_loss_curves(self, history: TrainingHistory) -> Path:
        """Plota as curvas de loss de treino e validação por época.

        Args:
            history: ``TrainingHistory`` retornado por ``TorchTrainer.train()``
                (``training_result.history``).

        Returns:
            Caminho do arquivo ``.png`` salvo.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.train_loss, label="Train Loss")
        if history.val_loss is not None:
            ax.plot(history.val_loss, label="Val Loss")
        ax.set_title("Training and Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()

        out_path = self._cfg.plots_dir / "loss_curves.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self._logger.info("Plot salvo: %s", out_path)
        return out_path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
    ) -> Path:
        """Plota e salva a matriz de confusão.

        Args:
            y_true:    Rótulos verdadeiros.
            y_pred:    Predições do modelo.
            normalize: Se ``True``, normaliza por linha (proporção por classe).

        Returns:
            Caminho do arquivo ``.png`` salvo.
        """
        viz_config = AssessmentVisualizationConfig(class_names=self._cfg.class_names)
        plotter = AssessmentVisualization(config=viz_config)
        plotter.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            normalize=normalize,
            title="Normalized Confusion Matrix" if normalize else "Confusion Matrix",
        )

        out_path = self._cfg.plots_dir / "confusion_matrix.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        self._logger.info("Plot salvo: %s", out_path)
        return out_path

    def plot_confusion_array(self, cm_norm: np.ndarray, title: str, filename: str) -> Path:
        """Plota uma matriz de confusão já normalizada por linha (linhas = real, colunas = predito)."""
        cm = np.asarray(cm_norm)
        names = self._cfg.class_names
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cm, vmin=0, vmax=1, cmap='Greens')
        ax.grid(False)
        ax.set_xticks(range(len(names)), labels=names, rotation=30, ha='right')
        ax.set_yticks(range(len(names)), labels=names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                        color='white' if cm[i, j] > 0.6 else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        fig.tight_layout()

        self._cfg.plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._cfg.plots_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self._logger.info("Plot salvo: %s", out_path)
        return out_path


def _jsonable(obj):
    """Converte tipos NumPy para JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class Pipeline:
    """
    Objetivo: 
        Orquestra as etapas do pipeline de ponta a ponta
        Cada etapa pode ser chamada individualmente ou em sequência via run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._setup_directories()
        self._setup_logging()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.data_processor  = DataProcessor(config)
        self.model_manager   = ModelManager(config)
        self.quantizer       = Quantizer(config)
        self.arduino_exporter = ArduinoExporter(config)
        self.plotter          = Plotter(config)

        # Estado interno populado pelas etapas
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test:  Optional[np.ndarray] = None
        self.y_test:  Optional[np.ndarray] = None

    def _setup_directories(self) -> None:
        """
        Objetivo:
            Cria a estrutura de pastas de saída caso não existam
        Retorno:
        
        """
        for d in (
            self._cfg.logs_dir,
            self._cfg.models_dir,
            self._cfg.sketches_dir,
            self._cfg.plots_dir
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """
        Objetivo: 
            Configura o ``logging`` para console e arquivo.
            ``force=True`` é necessário porque importar ``ThreeWToolkit`` já configura o root
            logger (via ``constants.py``/``setup_default_logging``) com seus próprios handlers;
            sem isso, ``basicConfig`` seria um no-op e ``session.log`` nunca seria criado.
        """
        log_file = self._cfg.logs_dir / "session.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
            force=True,
        )

    def step_load_data(self) -> None:
        """
        Objetivo:
            Carregamento, windowing, split e normalização do dataset
        Retorno:
        
        """
        data = self.data_processor.load_and_window()
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_processor.split(data)
        self.X_train, self.X_test = self.data_processor.normalize(self.X_train, self.X_test)

    def step_train(self) -> None:
        """
        Objetivo:
            Construção do trainer e treinamento do modelo
        Retorno:    
        """
        self.model_manager.build_trainer()
        self.model_manager.train(self.X_train, self.y_train)

    def step_evaluate(self) -> dict:
        """
        Objetivo:
            Avaliação FP32 com rastreamento de emissões na CPU do computador
        Returno:
            Dicionário com métricas e consumo energético
        """
        return self.model_manager.evaluate(self.X_test, self.y_test)

    def step_plot(self, eval_report: dict) -> None:
        """
        Objetivo:
            Geração e salvamento dos plots de treino e avaliação
        Argumentos: 
            eval_report: Dicionário retornado por step_evaluate()
        Retorno:
        """
        self.plotter.plot_loss_curves(self.model_manager.training_result.history)
        self.plotter.plot_confusion_matrix(
            y_true=eval_report['true_values'],
            y_pred=eval_report['predictions'],
            normalize=True,
        )

    def step_quantize_and_export(self) -> list[dict]:
        """
        Objetivo:
            Quantização, avaliação inteira no teste completo e geração dos sketches
            Para cada largura quantiza, avalia com a inferência inteira em todo o conjunto de teste,
            gera o sketch de amostras e salva as métricas
        Retorno:
        """
        X_export, y_export = ArduinoExporter.select_balanced_export(
            self.X_test, self.y_test,
            n=self._cfg.n_export,
            seed=self._cfg.random_seed,
        )
        self._logger.info(
            'Amostras exportadas - distribuição de classes: %s',
            {int(c): int((y_export == c).sum()) for c in np.unique(y_export)},
        )
        self._logger.info(
            'ENERGIA dos sketches é ESTIMADA: P = %.3f W constante (%.1f V x %.0f mA) x tempo',
            self._cfg.assumed_power_w, self._cfg.assumed_vcc, self._cfg.assumed_current_a * 1000,
        )

        model = self.model_manager.model
        results: list[dict] = []

        for bits in self._cfg.quantization_bits:
            q = self.quantizer.quantize_model(model, self.X_train, bits)
            metrics = self.quantizer.evaluate_quantized(
                q, model, self.X_test, self.y_test, n_max=self._cfg.eval_samples
            )
            metrics['overflow_warnings'] = q['overflow_warnings']
            self.quantizer.save(q, self._cfg.models_dir / f'quant_int{bits}.npz')
            self.plotter.plot_confusion_array(
                np.array(metrics['confusion_norm']), f'INT{bits}', f'confusion_int{bits}.png'
            )
            expected = self.quantizer.int_predict(q, X_export)
            self.arduino_exporter.export_sketch(q, X_export, y_export, expected_pred=expected)
            results.append(metrics)

        out = self._cfg.logs_dir / 'quantization_metrics.json'
        out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
        self._logger.info('Métricas salvas: %s', out)
        return results

    def run(self) -> None:
        self._logger.info('Iniciando pipeline...')
        self.step_load_data()
        self.step_train()
        report = self.step_evaluate()
        self.step_plot(report)
        self.step_quantize_and_export()
        self._logger.info('Pipeline finalizado.')

def main() -> None:
    config = PipelineConfig(
        dataset_path=r'C:\Users\joaov_zm1q2wh\projects\sbrt_2026_mts_classification_embedded_systems\data',
        window_size=1000,
        random_seed=42,
        epochs=100,
        quantization_bits=[8, 16, 32],
        n_export=10,
        output_dir=Path('./output'),
        split_mode='well', # event
    )
    set_global_seed(config.random_seed)
    Pipeline(config).run()

if __name__ == '__main__':
    main()