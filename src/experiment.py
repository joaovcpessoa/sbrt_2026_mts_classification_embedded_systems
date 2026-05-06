from __future__ import annotations

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

from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.preprocessing import Windowing
from ThreeWToolkit.models.mlp import MLP, MLPConfig
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import WindowingConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig
from ThreeWToolkit.assessment.assessment_visualizations import AssessmentVisualization
from ThreeWToolkit.core.base_assessment_visualization import AssessmentVisualizationConfig

# Mapeamentos de tipos C / NumPy por largura de bits
_NP_WEIGHT: dict[int, type] = {8: np.int8,  16: np.int16, 32: np.int32}
_NP_BIAS:   dict[int, type] = {8: np.int32, 16: np.int32, 32: np.int64}
_C_WEIGHT:  dict[int, str]  = {8: 'int8_t',  16: 'int16_t', 32: 'int32_t'}
_C_BIAS:    dict[int, str]  = {8: 'int32_t', 16: 'int32_t', 32: 'int64_t'}
_C_ACCUM:   dict[int, str]  = {8: 'int32_t', 16: 'int64_t', 32: 'int64_t'}

# Reproducibility
def set_global_seed(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade para reprodutibilidade total.

    Cobre: Python random, NumPy, PyTorch (CPU e CUDA) e flags de
    determinismo do cuDNN.

    Args:
        seed: Valor da semente.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# Configuração centralizada
@dataclass
class PipelineConfig:
    """Parâmetros globais do pipeline. Altere aqui para novos experimentos."""

    # Data
    dataset_path: str = r'C:\Users\jvt\Downloads\data\3W'
    selected_col: str = 'T-TPT'
    target_class: list[int] = field(default_factory=lambda: [3, 4])
    train_ratio: float = 0.8

    # Windowing
    window_size: int = 1000
    window_type: str = 'hann'
    label_window_type: str = 'boxcar'
    overlap: float = 0.5
    pad_last_window: bool = True

    # Seed
    random_seed: int = 2026

    # Model
    hidden_sizes: tuple[int, ...] = (32, 16)
    output_size: int = 5
    activation_function: str = 'relu'
    regularization: Optional[str] = None

    # Training
    optimizer: str = 'adam'
    criterion: str = 'cross_entropy'
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    cross_validation: bool = False
    shuffle_train: bool = True

    # Evaluation
    metrics: list[str] = field(default_factory=lambda: ['accuracy'])
    power_measure_secs: float = 0.1

    # Export
    quantization_bits: list[int] = field(default_factory=lambda: [8, 16, 32])
    n_export: int = 10
    calib_samples: int = 2000

    # Visualizações
    class_names: list[str] = field(default_factory=lambda: ['Normal Operation', 'Flow Instability', 'Severe Slugging'])

    # Output
    output_dir: Path = Path(r'/output')

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
class DataProcessor:
    """Carrega o dataset 3W, aplica windowing e divide em treino/teste."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)

    def load_and_window(self) -> pd.DataFrame:
        """Carrega eventos do dataset e aplica janelamento hann + boxcar.

        Returns:
            DataFrame com features e coluna ``label``.
        """
        self._logger.info('Carregando dataset de: %s', self._cfg.dataset_path)
        ds_config = ParquetDatasetConfig(
            path=self._cfg.dataset_path,
            clean_data=True,
            seed=self._cfg.random_seed,
            target_class=self._cfg.target_class,
        )
        ds = ParquetDataset(ds_config)

        windowing = Windowing(
            WindowingConfig(
                window=self._cfg.window_type,
                window_size=self._cfg.window_size,
                overlap=self._cfg.overlap,
                pad_last_window=self._cfg.pad_last_window,
            )
        )
        label_windowing = Windowing(
            WindowingConfig(
                window=self._cfg.label_window_type,
                window_size=self._cfg.window_size,
                overlap=self._cfg.overlap,
                pad_last_window=self._cfg.pad_last_window,
            )
        )

        dfs: list[pd.DataFrame] = []
        for event in ds:
            windowed_signal = windowing(event['signal'][self._cfg.selected_col])
            windowed_label  = label_windowing(event['label'])
            windowed_signal.drop(columns=['win'], inplace=True)
            windowed_signal['label'] = (
                windowed_label.drop(columns=['win']).mode(axis=1)[0].astype(int)
            )
            dfs.append(windowed_signal)

        combined = pd.concat(dfs, ignore_index=True, axis=0)
        self._logger.info('Windowing concluido: %d amostras totais', len(combined))
        return combined

    def split(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide o DataFrame em conjuntos de treino e teste.

        Args:
            df: DataFrame com features e coluna ``label``.

        Returns:
            Tupla ``(X_train, y_train, X_test, y_test)``.
        """
        n_total = len(df)
        n_train = int(self._cfg.train_ratio * n_total)
        rng = np.random.default_rng(self._cfg.random_seed)
        idx = rng.permutation(n_total)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_all = df.iloc[:, :-1].values.astype(np.float32)
        y_all = df['label'].values.astype(np.int64)

        self._logger.info('Split: %d treino / %d teste', len(train_idx), len(test_idx))

        return (X_all[train_idx], y_all[train_idx], X_all[test_idx],  y_all[test_idx])

# ModelManager
class ModelManager:
    """Responsável pelo treinamento e avaliação do modelo MLP."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.trainer: Optional[ModelTrainer] = None

    def build_trainer(self) -> ModelTrainer:
        """Instancia o ``ModelTrainer`` com os hiperparâmetros da config."""
        mlp_config = MLPConfig(
            input_size=self._cfg.window_size,
            hidden_sizes=self._cfg.hidden_sizes,
            output_size=self._cfg.output_size,
            random_seed=self._cfg.random_seed,
            activation_function=self._cfg.activation_function,
            regularization=self._cfg.regularization,
        )
        trainer_config = TrainerConfig(
            optimizer=self._cfg.optimizer,
            criterion=self._cfg.criterion,
            batch_size=self._cfg.batch_size,
            epochs=self._cfg.epochs,
            seed=self._cfg.random_seed,
            config_model=mlp_config,
            learning_rate=self._cfg.learning_rate,
            cross_validation=self._cfg.cross_validation,
            shuffle_train=self._cfg.shuffle_train,
        )
        self.trainer = ModelTrainer(trainer_config)
        return self.trainer

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Treina o modelo.

        Args:
            X_train: Features de treino.
            y_train: Rótulos de treino.
        """
        if self.trainer is None:
            self.build_trainer()
        
        self._logger.info('Iniciando treinamento (%d epocas).', self._cfg.epochs)
        self.trainer.train(x_train=pd.DataFrame(X_train),  y_train=pd.Series(y_train))
        self._logger.info('Treinamento concluido.')

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Avalia o modelo com rastreamento de emissões de CO₂.

        Args:
            X_test: Features de teste.
            y_test: Rótulos de teste.

        Returns:
            Dicionário com métricas, tempos e consumo energético.
        """
        assessment_config = ModelAssessmentConfig( metrics=self._cfg.metrics, batch_size=self._cfg.batch_size, output_dir=self._cfg.output_dir)
        tracker = EmissionsTracker(measure_power_secs=self._cfg.power_measure_secs, save_to_file=True, output_dir=self._cfg.output_dir)

        t0 = time.time()
        tracker.start()
        results = self.trainer.assess(
            pd.DataFrame(X_test),
            pd.Series(y_test.astype(int)),
            assessment_config=assessment_config,
        )
        tracker.stop()
        duration = time.time() - t0

        energy_kwh    = tracker.final_emissions_data.energy_consumed
        energy_wh     = energy_kwh * 1000
        energy_joules = energy_wh * 3600
        n_samples     = len(X_test)
        emissions_kg  = tracker.final_emissions_data.emissions

        report = {
            "metrics":          results["metrics"],
            "duration_s":       duration,
            "n_samples":        n_samples,
            "energy_joules":    energy_joules,
            "energy_wh":        energy_wh,
            "co2_kg":           emissions_kg,
            "us_per_sample":    duration / n_samples * 1e6,
            "uj_per_sample":    energy_joules / n_samples * 1e6,
            "true_values":      results["true_values"],
            "predictions":      results["predictions"]
        }

        self._logger.info('Métricas de teste        : %s', report["metrics"])
        self._logger.info('Tempo total de inferência (FP32)  : %.3f s', report["duration_s"])
        self._logger.info('Tempo médio por amostra (FP32)    : %.2f µs', report["us_per_sample"])
        self._logger.info('Energia consumida (FP32)          : %.4f J  /  %.6f Wh', report["energy_joules"], report["energy_wh"])
        self._logger.info('Energia média por amostra (FP32)  : %.4f µJ', report["uj_per_sample"])
        self._logger.info('CO₂ estimado                      : %.6f kg', report["co2_kg"])
        return report

    @property
    def model(self) -> MLP:
        """Retorna o modelo treinado."""
        return self.trainer.model

# Quantizer
class Quantizer:
    """Quantização pós-treinamento (PTQ) para INT8, INT16 e INT32."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)

    # helpers estáticos (lógica preservada integralmente)

    @staticmethod
    def _get_all_linear_layers(
        mlp: MLP,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extrai pesos e biases de todas as camadas ``nn.Linear`` do MLP.

        Args:
            mlp: Modelo MLP treinado.

        Returns:
            Lista de tuplas ``(W, b)`` com arrays NumPy.
        """
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        for module in mlp.model:
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy().T  # [in, out]
                b = module.bias.detach().cpu().numpy()      # [out]
                layers.append((W, b))
        return layers

    @staticmethod
    def _forward_numpy(
        mlp: MLP, x_np: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Inferência em NumPy capturando ativações intermediárias.

        Args:
            mlp:  Modelo MLP.
            x_np: Amostra de entrada (array 1D).

        Returns:
            Tupla ``(hidden_activations, logits)``.
        """
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
            hidden_acts: list[np.ndarray] = []
            for module in mlp.model:
                x = module(x)
                if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
                    hidden_acts.append(x.squeeze().cpu().numpy().copy())
            logits = x.squeeze().cpu().numpy().copy()
        return hidden_acts, logits

    @staticmethod
    def _calc_scale(arr: np.ndarray, bits: int) -> float:
        """Escala de quantização baseada no valor absoluto máximo.

        Args:
            arr:  Array de referência.
            bits: Largura de bits (8, 16 ou 32).

        Returns:
            Escala como float.
        """
        return float(np.max(np.abs(arr))) / (2 ** (bits - 1) - 1) + 1e-8

    @staticmethod
    def quantize_multiplier(m: float) -> tuple[int, int]:
        """Converte um multiplicador real em (mult, shift) inteiro Q31.

        Args:
            m: Multiplicador em ponto flutuante.

        Returns:
            Tupla ``(mult, shift)`` para aritmética inteira.
        """
        if m == 0.0:
            return 0, 0
        shift = 0
        while m < 0.5 and shift < 31:
            m *= 2
            shift += 1
        mult = min(int(np.round(m * (1 << 31))), (1 << 31) - 1)
        return mult, shift

    @staticmethod
    def fp_requant(
        acc: int, mult: int, shift: int, out_min: int, out_max: int
    ) -> int:
        """Re-quantização de ponto fixo com clipping.

        Args:
            acc:     Acumulador inteiro.
            mult:    Multiplicador Q31.
            shift:   Deslocamento de bits.
            out_min: Limite inferior.
            out_max: Limite superior.

        Returns:
            Valor re-quantizado e clampado.
        """
        x = (int(acc) * mult) >> 31
        x >>= shift
        return max(out_min, min(out_max, x))

    @staticmethod
    def dequantize(q_int: np.ndarray, scale: float) -> np.ndarray:
        """Dequantiza um array inteiro para float64.

        Args:
            q_int: Array de inteiros quantizados.
            scale: Escala de quantização.

        Returns:
            Array float64 dequantizado.
        """
        return q_int.astype(np.float64) * scale

    # método principal

    def quantize_model(self, mlp: MLP, X_calib: np.ndarray, bits: int) -> dict:
        """Realiza a quantização completa do modelo para ``bits`` bits.

        Utiliza um subconjunto de calibração para estimar escalas de ativação
        e avalia divergência, SNR e RMSE em relação ao modelo FP32.

        Args:
            mlp:     Modelo MLP treinado (em modo ``eval``).
            X_calib: Conjunto de calibração (features de treino).
            bits:    Largura de bits alvo (8, 16 ou 32).

        Returns:
            Dicionário com pesos quantizados, escalas e multiplicadores.
        """
        self._logger.info("Quantizando modelo – INT%d", bits)

        layer_params = self._get_all_linear_layers(mlp)
        n_layers = len(layer_params)
        s_W = [self._calc_scale(W, bits) for W, _ in layer_params]

        calib_idx = np.random.choice(
            len(X_calib), size=min(self._cfg.calib_samples, len(X_calib)), replace=False
        )

        clip_max = 2 ** (bits - 1) - 1
        clip_min = -(2 ** (bits - 1))

        x_abs_max: list[float] = []
        h_all: list[list[np.ndarray]] = [[] for _ in range(n_layers - 1)]
        logits_all: list[np.ndarray] = []

        mlp.eval()
        for i in calib_idx:
            x_abs_max.append(float(np.max(np.abs(X_calib[i]))))
            hidden_acts, logits = self._forward_numpy(mlp, X_calib[i])
            for l, h in enumerate(hidden_acts):
                h_all[l].append(h)
            logits_all.append(logits)

        s_x      = float(np.max(x_abs_max)) / clip_max + 1e-8
        s_h      = [self._calc_scale(np.array(h_list), bits) for h_list in h_all]
        s_logits = self._calc_scale(np.array(logits_all), bits)

        s_in  = [s_x] + s_h
        s_out = s_h + [s_logits]

        W_q, b_q, MULTS, SHIFTS = [], [], [], []
        for l, (W, b) in enumerate(layer_params):
            W_q.append(
                np.clip(np.round(W / s_W[l]), clip_min, clip_max).astype(_NP_WEIGHT[bits])
            )
            b_q.append(
                np.round(b / (s_in[l] * s_W[l])).astype(_NP_BIAS[bits])
            )
            M = (s_in[l] * s_W[l]) / s_out[l]
            mult, shift = self.quantize_multiplier(M)
            MULTS.append(mult)
            SHIFTS.append(shift)

        def calc_snr(orig: np.ndarray, quant: np.ndarray) -> float:
            noise = orig - quant
            pn = np.mean(noise ** 2)
            return 10.0 * np.log10(np.mean(orig ** 2) / pn) if pn > 0 else np.inf

        errors, snr_vals, rmse_vals = [], [], []
        for i in calib_idx[:200]:
            _, lf = self._forward_numpy(mlp, X_calib[i])
            h = np.clip(np.round(X_calib[i] / s_x), clip_min, clip_max).astype(np.int64)
            for l in range(n_layers):
                z = h @ W_q[l].astype(np.int64) + b_q[l].astype(np.int64)
                h_req = np.array(
                    [self.fp_requant(int(v), MULTS[l], SHIFTS[l], clip_min, clip_max) for v in z]
                )
                if l < n_layers - 1:
                    h = np.maximum(0, h_req).astype(np.int64)
                else:
                    h = h_req
            lq_dequant = self.dequantize(h, s_logits)
            errors.append(int(np.argmax(lf) != int(np.argmax(h))))
            snr_vals.append(calc_snr(lf, lq_dequant))
            rmse_vals.append(float(np.sqrt(np.mean((lf - lq_dequant) ** 2))))

        flash_kb = sum(W.nbytes + b.nbytes for W, b in zip(W_q, b_q)) / 1024
        self._logger.info(
            "INT%d – Divergência FP32 vs INT%d: %.1f%%  |  SNR: %.2f dB  |  "
            "RMSE: %.6f  |  Flash: ~%.1f KB",
            bits, bits, np.mean(errors) * 100,
            np.nanmean(snr_vals), np.mean(rmse_vals), flash_kb,
        )

        return dict(
            bits=bits,
            n_layers=n_layers,
            input_size=self._cfg.window_size,
            hidden_sizes=self._cfg.hidden_sizes,
            output_size=self._cfg.output_size,
            W_q=W_q, b_q=b_q,
            s_x=s_x, s_h=s_h, s_logits=s_logits,
            MULTS=MULTS, SHIFTS=SHIFTS,
        )

# ArduinoExporter
class ArduinoExporter:
    """Gera sketches Arduino (.ino) a partir de modelos quantizados."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._logger = logging.getLogger(self.__class__.__name__)

    # ── helpers internos ────────────────────────────────────────────────

    @staticmethod
    def _c_array(name: str, arr: np.ndarray, c_type: str, per_line: int = 16) -> str:
        """Serializa um array NumPy como array C/PROGMEM.

        Args:
            name:     Nome da variável C.
            arr:      Array NumPy a serializar.
            c_type:   Tipo C (ex: ``int8_t``).
            per_line: Valores por linha no array C.

        Returns:
            String com a declaração do array C.
        """
        flat = [int(v) for v in arr.flatten()]
        rows = [
            "  " + ", ".join(map(str, flat[i : i + per_line]))
            for i in range(0, len(flat), per_line)
        ]
        return f"const {c_type} {name}[] PROGMEM = {{\n" + ",\n".join(rows) + "\n};\n\n"

    @staticmethod
    def _pgm_read_weight(bits: int, arr_name: str, flat_idx: str) -> str:
        """Gera a macro de leitura PROGMEM para pesos.

        Args:
            bits:     Largura de bits.
            arr_name: Nome do array C.
            flat_idx: Expressão do índice.

        Returns:
            Expressão C para leitura de peso em PROGMEM.
        """
        if bits == 8:
            return f"((int32_t)(int8_t) pgm_read_byte (&{arr_name}[{flat_idx}]))"
        if bits == 16:
            return f"((int64_t)(int16_t)pgm_read_word (&{arr_name}[{flat_idx}]))"
        return f"((int64_t)(int32_t)pgm_read_dword(&{arr_name}[{flat_idx}]))"

    @staticmethod
    def _pgm_read_bias(bits: int, arr_name: str, idx: str) -> str:
        """Gera a macro de leitura PROGMEM para biases."""
        if bits == 8:
            return f"((int32_t)         pgm_read_dword(&{arr_name}[{idx}]))"
        if bits == 16:
            return f"((int64_t)(int32_t)pgm_read_dword(&{arr_name}[{idx}]))"
        return f"read_int64_p(&{arr_name}[{idx}])"

    @staticmethod
    def _pgm_read_input(bits: int, arr_name: str, idx: str) -> str:
        """Gera a macro de leitura PROGMEM para entradas."""
        if bits == 8:
            return f"((int8_t)          pgm_read_byte (&{arr_name}[{idx}]))"
        if bits == 16:
            return f"((int16_t)         pgm_read_word (&{arr_name}[{idx}]))"
        return f"((int32_t)         pgm_read_dword(&{arr_name}[{idx}]))"

    # ── construção das seções do sketch ─────────────────────────────────

    def _build_defines(self, q: dict, N: int) -> str:
        """Gera a seção de ``#define`` do sketch."""
        bits = q["bits"]
        hidden_sizes = q["hidden_sizes"]
        defines = (
            "#include <Arduino.h>\n"
            "#include <avr/pgmspace.h>\n"
            "#include <string.h>\n\n"
            f"#define INPUT_SIZE  {q['input_size']}\n"
        )
        for l, h in enumerate(hidden_sizes):
            defines += f"#define HIDDEN{l + 1}     {h}\n"
        defines += f"#define OUTPUT_SIZE {q['output_size']}\n"
        defines += f"#define N_TEST      {N}\n\n"
        for l in range(q["n_layers"]):
            defines += f"#define MULT{l + 1}  ((int32_t){q['MULTS'][l]}L)\n"
            defines += f"#define SHIFT{l + 1} {q['SHIFTS'][l]}\n"
        defines += "\n"
        return defines

    def _build_data_section(self, q: dict, X_te_q: np.ndarray, y_te: np.ndarray, N: int) -> str:
        """Gera os arrays PROGMEM de pesos, biases e amostras de teste."""
        bits = q["bits"]
        cw = _C_WEIGHT[bits]
        cb = _C_BIAS[bits]
        data_sec = ""
        for l in range(q["n_layers"]):
            data_sec += self._c_array(f"W{l + 1}_q", q["W_q"][l], cw)
            data_sec += self._c_array(f"b{l + 1}_q", q["b_q"][l], cb)
        data_sec += f"// {N} amostras de teste\n"
        data_sec += self._c_array("X_test_q", X_te_q, cw)
        data_sec += self._c_array("y_labels", y_te[:N].astype(np.uint8), "uint8_t")
        return data_sec

    def _build_int64_helper(self, bits: int) -> str:
        """Gera o helper de leitura int64 (apenas necessário para INT32)."""
        if bits != 32:
            return ""
        return (
            "static int64_t read_int64_p(const int64_t *addr) {\n"
            "  int64_t v;\n"
            "  memcpy_P(&v, addr, sizeof(v));\n"
            "  return v;\n"
            "}\n\n"
        )

    def _build_requant_block(self, bits: int) -> str:
        """Gera a função ``requant`` em C para o número de bits."""
        if bits == 8:
            return (
                "static int8_t requant(int32_t acc, int32_t mult, int shift) {\n"
                "  int64_t x = (int64_t)acc * mult;\n"
                "  x >>= 31;\n"
                "  x >>= shift;\n"
                "  if (x >  127) x =  127;\n"
                "  if (x < -128) x = -128;\n"
                "  return (int8_t)x;\n"
                "}\n\n"
            )
        out_type = "int16_t" if bits == 16 else "int32_t"
        hi_val   = "32767"   if bits == 16 else "2147483647LL"
        lo_val   = "-32768"  if bits == 16 else "-2147483648LL"
        return (
            f"static int64_t mulq31(int64_t a, int32_t m) {{\n"
            f"  int64_t a_hi = a >> 31;\n"
            f"  int64_t a_lo = a & 0x7FFFFFFFL;\n"
            f"  return (int64_t)a_hi * m + (((int64_t)a_lo * m) >> 31);\n"
            f"}}\n\n"
            f"static {out_type} requant(int64_t acc, int32_t mult, int shift) {{\n"
            f"  int64_t x = mulq31(acc, mult);\n"
            f"  x >>= shift;\n"
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
            cur_size  = layer_sizes[l + 1]
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
                comment += "  (raw accumulator – argmax invariant to scale)"

            rw = self._pgm_read_weight(bits, W_name, f"i * {size_macro} + j")
            rb = self._pgm_read_bias(bits, b_name, "j")
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

    def _build_setup_loop(self, q: dict) -> str:
        """Gera as funções ``setup()`` e ``loop()`` do sketch."""
        bits = q["bits"]
        cw   = _C_WEIGHT[bits]
        rx_test = self._pgm_read_input(bits, "X_test_q", "n * INPUT_SIZE + i")
        return (
            "const float VCC      = 5.0;\n"
            "const float I_ACTIVE = 0.020;\n"
            "const float POWER_W  = VCC * I_ACTIVE;\n\n"
            f"void setup() {{\n"
            "  Serial.begin(115200);\n"
            f'  Serial.println("Validacao MLP INT{bits}");\n\n'
            "  int correct = 0;\n"
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
            "    int label = (int)pgm_read_byte(&y_labels[n]);\n"
            "    if (pred == label) correct++;\n\n"
            '    Serial.print("Amostra "); Serial.print(n);\n'
            '    Serial.print(" | Pred: ");    Serial.print(pred);\n'
            '    Serial.print(" | Real: ");    Serial.print(label);\n'
            '    Serial.print(" | Tempo: ");   Serial.print(duration_us);   Serial.print("us");\n'
            '    Serial.print(" | Energia: "); Serial.print(energy_joules, 10); Serial.println(" J");\n'
            "  }\n\n"
            "  float avg_time_us = (float)total_time_us / N_TEST;\n"
            "  float avg_joules  = total_joules / N_TEST;\n\n"
            '  Serial.println("------------------------------------------");\n'
            '  Serial.print("Acuracia Final: "); Serial.print(correct); Serial.print("/"); Serial.println(N_TEST);\n'
            '  Serial.print("Tempo Medio por Amostra: "); Serial.print(avg_time_us); Serial.println(" us");\n'
            '  Serial.print("Energia Media por Amostra: "); Serial.print(avg_joules, 10); Serial.println(" J");\n'
            '  Serial.print("Tempo Total: "); Serial.print(total_time_us / 1000.0); Serial.println(" ms");\n'
            '  Serial.print("Energia Total: "); Serial.print(total_joules, 8); Serial.println(" J");\n'
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
    ) -> Path:
        """Gera e salva o sketch Arduino para um modelo quantizado.

        Args:
            q:    Dicionário de quantização retornado por ``Quantizer.quantize_model``.
            X_te: Amostras de teste para embutir no sketch.
            y_te: Rótulos correspondentes.

        Returns:
            Caminho do arquivo ``.ino`` gerado.
        """
        bits = q["bits"]
        clip_max = 2 ** (bits - 1) - 1
        clip_min = -(2 ** (bits - 1))
        N = min(self._cfg.n_export, len(X_te))

        X_te_q = np.clip(
            np.round(X_te[:N] / q["s_x"]), clip_min, clip_max
        ).astype(_NP_WEIGHT[bits])

        sketch = "".join([
            self._build_defines(q, N),
            self._build_data_section(q, X_te_q, y_te, N),
            self._build_int64_helper(bits),
            self._build_requant_block(bits),
            self._build_infer_body(q),
            self._build_setup_loop(q),
        ])

        out_path = self._cfg.sketches_dir / f"mlp_3w_int{bits}.ino"
        out_path.write_text(sketch, encoding="utf-8")

        flash_kb = sum(W.nbytes + b.nbytes for W, b in zip(q["W_q"], q["b_q"])) / 1024
        self._logger.info(
            "Sketch gerado: %s  |  Flash ~%.1f KB  |  %d amostras de teste",
            out_path, flash_kb, N,
        )
        return out_path

    @staticmethod
    def select_balanced_export(
        X: np.ndarray, y: np.ndarray, n: int, seed: int = 2026
    ) -> tuple[np.ndarray, np.ndarray]:
        """Seleciona ``n`` amostras balanceadas entre as classes.

        Garante ≈ ``n // n_classes`` amostras por classe; vagas restantes
        são preenchidas pela classe majoritária.

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
        base    = n // n_cls
        extra   = n % n_cls

        chosen: list[int] = []
        for i, c in enumerate(classes):
            idx_c  = np.where(y == c)[0]
            n_pick = min(base + (1 if i < extra else 0), len(idx_c))
            chosen.extend(rng.choice(idx_c, size=n_pick, replace=False).tolist())

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

    def plot_loss_curves(self, trainer: ModelTrainer) -> Path:
        """Plota as curvas de loss de treino e validação por época.
 
        Lê ``trainer.history[0]`` para extrair ``train_loss`` e ``val_loss``.
 
        Args:
            trainer: ``ModelTrainer`` após o treinamento.
 
        Returns:
            Caminho do arquivo ``.png`` salvo.
        """
        history = trainer.history[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["train_loss"], label="Train Loss")
        ax.plot(history["val_loss"],   label="Val Loss")
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

# Pipeline
class Pipeline:
    """Orquestra as etapas do pipeline de ponta a ponta.

    Cada etapa pode ser chamada individualmente ou em sequência via ``run()``.
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
        """Cria a estrutura de pastas de saída caso não existam."""
        for d in (
            self._cfg.logs_dir,
            self._cfg.models_dir,
            self._cfg.sketches_dir,
            self._cfg.plots_dir
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Configura o ``logging`` para console e arquivo."""
        log_file = self._cfg.logs_dir / "session.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
        )

    # etapas individuais
    def step_load_data(self) -> None:
        """Etapa 1: Carregamento, windowing e split do dataset."""
        df = self.data_processor.load_and_window()
        self.X_train, self.y_train, self.X_test, self.y_test = (self.data_processor.split(df))

    def step_train(self) -> None:
        """Etapa 2: Construção do trainer e treinamento do modelo."""
        self.model_manager.build_trainer()
        self.model_manager.train(self.X_train, self.y_train)

    def step_evaluate(self) -> dict:
        """Etapa 3: Avaliação FP32 com rastreamento de emissões.

        Returns:
            Dicionário com métricas e consumo energético.
        """
        return self.model_manager.evaluate(self.X_test, self.y_test)
    
    def step_plot(self, eval_report: dict) -> None:
        """Etapa 4: Geração e salvamento dos plots de treino e avaliação.
 
        Salva ``loss_curves.png`` e ``confusion_matrix.png`` em
        ``outputs/plots/``.
 
        Args:
            eval_report: Dicionário retornado por ``step_evaluate()``.
        """
        self.plotter.plot_loss_curves(self.model_manager.trainer)
        self.plotter.plot_confusion_matrix(
            y_true=eval_report["true_values"],
            y_pred=eval_report["predictions"],
            normalize=True,
        )

    def step_quantize_and_export(self) -> None:
        """Etapa 4: Quantização e geração dos sketches Arduino."""
        X_export, y_export = ArduinoExporter.select_balanced_export(
            self.X_test, self.y_test,
            n=self._cfg.n_export,
            seed=self._cfg.random_seed,
        )
        self._logger.info(
            "Amostras exportadas – distribuição de classes: %s",
            {int(c): int((y_export == c).sum()) for c in np.unique(y_export)},
        )

        model = self.model_manager.model
        model.eval()

        for bits in self._cfg.quantization_bits:
            q = self.quantizer.quantize_model(model, self.X_train, bits)
            self.arduino_exporter.export_sketch(q, X_export, y_export)

    # execução completa
    def run(self) -> None:
        """Executa o pipeline completo em sequência."""
        self._logger.info("Iniciando pipeline")
        self.step_load_data()
        self.step_train()
        report = self.step_evaluate()
        self.step_plot(report)
        self.step_quantize_and_export()
        self._logger.info("Pipeline finalizado")

# Main
def main() -> None:
    config = PipelineConfig(
        dataset_path=r'C:\Users\jvt\Downloads\data\3W',
        window_size=1000,
        random_seed=42,
        epochs=100,
        quantization_bits=[8, 16, 32],
        n_export=10,
        output_dir=Path('./output'),
    )
    set_global_seed(config.random_seed)
    Pipeline(config).run()

# main()