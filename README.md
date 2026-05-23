# Quantized MLP inference in MCU

## Sumário

- [Objetivo](#objetivo)
- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Configuração](#configuração)
- [Uso](#uso)
- [Saídas Geradas](#saídas-geradas)
- [Detalhes Técnicos](#detalhes-técnicos)
- [Referências](#referências)

## Objetivo

Comparar características de uma MLP implementada para um problema de classificação com:
- Ponto flutuante (FP32)
- Ponto fixo (INT8, INT16, INT32)
Avaliação:
- Tempo de inferência
- Uso de memória
- Erro numérico
- Precisão da classificação
- Energia

### Representação em ponto fixo

Use quantização uniforme:

$$x_q = \text{round}(x \cdot 2^n)$$
​
$$x \approx \frac{x_q}{2^n}$$

Onde $n$ = número de bits fracionários.

Formatos de teste:

| Formato | Comprimento | Bits |
| ------ | ----------- | ---------------- | 
| Q7	   | 8 bits	     | 7                | 
| Q15	   | 16 bits     | 15               | 
| Q31	   | 32 bits     | 31               |

### Estratégia de Quantização

Escalonamento por camada:

$$S=\frac{2^{n−1}-1}{max(∣x∣)}$$

$$x_q=round(S \cdot x)$$

Isso evita estouro de capacidade e é padrão em redes neurais quantizadas.

### O que será medido?

**Acurácia**

$$ Acurácia = \frac{\text{correct predictions}}{\text{total samples}} $$
​
**Erro**

Comparar a saída em ponto flutuante com a saída em ponto fixo:

$$RMSE=\sqrt{\frac{1}{N} \sum (y_{float} - y_{fixed})^2}$$

$$SNR=10log10 (\frac{\sum y^2_{float}}{\sum (y_{float} - y_{fixed})^2})$$
 
**Tempo de inferência**

$$ \text{Tempo de inferência} = \frac{\text{Total time}}{\text{Number of samples}} $$

**Uso de memória**

$$ \text{Memória} = \text{weights} + \text{activations}$$

A ideia para o código foi criar um pipeline modular em Python capaz de executar essas tarefas de forma ordenada e fácil de compreender. Nosso objetivo principal foi a detecção de eventos em poços de petróleo usando o dataset 3W, contruído pela Petrobrás. O projeto segue o seguinte fluxo: 
- pré-processamento com windowing
- treinamento de uma MLP
- avaliação das medidas
- quantização pós-treinamento (INT8/INT16/INT32)
- geração automática de sketches para microcontroladores AVR



## Visão Geral

O dataset 3W contém séries temporais de sensores de poços de petróleo rotuladas com eventos anômalos. Este pipeline:

1. Carrega e limpa os dados via `ThreeWToolkit`, criado pela Petrobrás para manipular o dataset
2. Aplica janelamento espectral (Hann) no sinal e janelamento Boxcar nos rótulos
3. Treina uma MLP com PyTorch
4. Avalia acurácia e mede consumo energético (Joules/Wh) e emissões de CO₂ com a biblioteca CodeCarbon
5. Quantiza o modelo treinado em INT8, INT16 e INT32 usando aritmética de ponto fixo Q31
6. Exporta sketches `.ino` autocontidos para validação embarcada em placas Arduino AVR

## Arquitetura

O código é organizado em seis classes com responsabilidades bem definidas:

```txt
PipelineConfig          ← Configuração centralizada (dataclass)
│
├── DataProcessor       ← Carregamento, windowing e split treino/teste
├── ModelManager        ← Treinamento e avaliação FP32 com rastreamento de CO₂
├── Quantizer           ← Quantização PTQ (INT8 / INT16 / INT32)
├── ArduinoExporter     ← Geração de sketches .ino para AVR
│
└── Pipeline            ← Orquestrador: coordena todas as etapas
```

### Fluxo de execução

```txt
Dataset 3W (Parquet)
        │
        ▼
  DataProcessor
  ┌─────────────────────────────────┐
  │  Windowing Hann  →  Features    │
  │  Windowing Boxcar → Labels      │
  │  Train / Test split (80/20)     │
  └─────────────────────────────────┘
        │
        ▼
  ModelManager
  ┌─────────────────────────────────┐
  │  Treina MLP (PyTorch)           │
  │  Avalia com EmissionsTracker    │
  │  → Acurácia, Energia, CO₂      │
  └─────────────────────────────────┘
        │
        ▼
  Quantizer  (para cada largura de bits)
  ┌─────────────────────────────────┐
  │  Calibração de escalas          │
  │  Quantização de pesos (Q31)     │
  │  Avaliação: SNR, RMSE, Flash    │
  └─────────────────────────────────┘
        │
        ▼
  ArduinoExporter
  ┌─────────────────────────────────┐
  │  Gera mlp_3w_int{8,16,32}.ino  │
  │  Com PROGMEM, requant, argmax   │
  └─────────────────────────────────┘
        │
        ▼
  outputs/
  ├── logs/session.log
  ├── models/
  └── sketches/mlp_3w_int{8,16,32}.ino
```

## Pré-requisitos

- Python 3.10+
- PyTorch 2.x
- Dataset 3W baixado localmente ([Petrobras/3W no GitHub](https://github.com/petrobras/3W))

### Dependências Python

```
numpy
pandas
torch
codecarbon
ThreeWToolkit
```

> `ThreeWToolkit` é a biblioteca oficial de utilitários do dataset 3W.  
> Consulte o repositório do projeto para instruções de instalação.

## Instalação

```bash
# 1. Clone este repositório
git clone https://github.com/seu-usuario/quantized_mlp_inference_mcu_sbrt_2026.git
cd quantized_mlp_inference_mcu_sbrt_2026

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install numpy pandas torch codecarbon

# 4. Instale o ThreeWToolkit (siga as instruções do repositório oficial)
```

## Estrutura de Diretórios

```txt
quantized_mlp_inference_mcu_sbrt_2026/
│
├── experiment.py      ← Código principal
├── experiment.ipynb   ← Jupyter Notebook
├── README.md
│
└── outputs/                          ← Criado automaticamente na primeira execução
    ├── logs/
    │   └── session.log               ← Log completo da sessão
    ├── models/                       ← Reservado para checkpoints futuros
    ├── plots/                        ← Reservado para os imagens
    └── sketches/
        ├── mlp_3w_int8.ino
        ├── mlp_3w_int16.ino
        └── mlp_3w_int32.ino
```

## Configuração

Toda a parametrização do experimento está centralizada na dataclass `PipelineConfig`. Para um novo experimento, basta alterar os valores na função `main()` — sem tocar no restante do código.

```python
config = PipelineConfig(
    dataset_path=r"/caminho/para/3W", # Caminho para o dataset 3W
    selected_col="T-TPT",             # Coluna de sinal utilizada
    target_class=[1, 2],              # Classes de eventos alvo (1 = Abrupt BSW Increase, 2 = Spurious Closure of DHSV)
    # Janelamento
    window_size=1000,        # Número de amostras por janela
    window_type="hann",      # Janela espectral para o sinal
    overlap=0.5,             # Sobreposição entre janelas (50%)
    # Modelo
    hidden_sizes=(32, 16),   # Neurônios por camada oculta
    output_size=3,           # Número de classes de saída
    epochs=20,
    learning_rate=0.001,
    # Quantização e exportação
    quantization_bits=[8, 16, 32],
    n_export=10,             # Amostras embarcadas no sketch Arduino
    # Diretório raiz dos outputs
    output_dir=Path("./outputs"),
)
```

### Parâmetros principais

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `dataset_path` | — | Caminho local para o dataset 3W |
| `selected_col` | `"T-TPT"` | Coluna de sinal do evento |
| `target_class` | `[1, 2]` | Classes de eventos a incluir |
| `window_size` | `1000` | Tamanho da janela temporal |
| `overlap` | `0.5` | Sobreposição entre janelas |
| `hidden_sizes` | `(32, 16)` | Arquitetura da MLP (camadas ocultas) |
| `epochs` | `20` | Épocas de treinamento |
| `learning_rate` | `0.001` | Taxa de aprendizado (Adam) |
| `quantization_bits` | `[8, 16, 32]` | Larguras de bits para quantização |
| `n_export` | `10` | Amostras exportadas para o sketch Arduino |
| `calib_samples` | `2000` | Amostras usadas na calibração da quantização |
| `random_seed` | `2026` | Semente global para reprodutibilidade |

## Uso

### Execução completa do pipeline

```bash
python experiment.py
```

### Execução por etapas (modo programático)

```python
from pathlib import Path
from experiment import Pipeline, PipelineConfig

config = PipelineConfig(dataset_path="/dados/3W", epochs=30)
pipeline = Pipeline(config)

# Execute etapas individualmente conforme necessário
pipeline.step_load_data()
pipeline.step_train()
report = pipeline.step_evaluate()
pipeline.step_quantize_and_export()
```

### Apenas exportar para Arduino (com modelo já treinado)

```python
from experiment import ArduinoExporter, Quantizer, PipelineConfig

config = PipelineConfig(...)
quantizer = Quantizer(config)
exporter  = ArduinoExporter(config)

q = quantizer.quantize_model(trained_model, X_train, bits=8)
exporter.export_sketch(q, X_test, y_test)
```

## Saídas Geradas

### Log de sessão (`outputs/logs/session.log`)

Registra todas as etapas com timestamp, incluindo métricas de avaliação FP32:

```
2026-05-01 14:32:10  INFO      ModelManager – Métricas de teste        : {'accuracy': 0.9412}
2026-05-01 14:32:10  INFO      ModelManager – Tempo total de inferência (FP32)  : 1.847 s
2026-05-01 14:32:10  INFO      ModelManager – Energia consumida (FP32)          : 0.0231 J  /  0.000006 Wh
2026-05-01 14:32:10  INFO      ModelManager – CO₂ estimado              : 0.000003 kg
```

### Relatório de quantização (no log)

Para cada largura de bits:

```
INFO  Quantizer – INT8 – Divergência FP32 vs INT8: 2.5%  |  SNR: 38.74 dB  |  RMSE: 0.000412  |  Flash: ~2.1 KB
```

### Sketches Arduino (`outputs/sketches/`)

Arquivos `.ino` autocontidos com:
- Pesos e biases quantizados em `PROGMEM`
- Amostras de teste balanceadas por classe
- Função `mlp_predict()` com aritmética inteira Q31
- Medição de tempo (`micros()`) e energia por inferência via `setup()`

## Detalhes Técnicos

### Quantização pós-treinamento (PTQ)

A quantização utiliza escalonamento por valor absoluto máximo (*absmax*) com fator de regularização ε = 1e-8 para evitar divisão por zero. O multiplicador de re-quantização é representado no formato Q31 (31 bits fracionários), garantindo aritmética inteira pura no microcontrolador.

Tipos por largura de bits:

| Bits | Pesos (`W`) | Biases (`b`) | Acumulador |
|------|-------------|--------------|------------|
| 8    | `int8_t`    | `int32_t`    | `int32_t`  |
| 16   | `int16_t`   | `int32_t`    | `int64_t`  |
| 32   | `int32_t`   | `int64_t`    | `int64_t`  |

### Métricas de qualidade da quantização

- **Divergência**: percentual de amostras em que `argmax(FP32) ≠ argmax(INTn)` — avalia impacto real na classificação.
- **SNR** (Signal-to-Noise Ratio): relação sinal-ruído entre logits FP32 e dequantizados, em dB.
- **RMSE**: erro quadrático médio entre os logits originais e dequantizados.

### Geração de energia no Arduino

O sketch embarcado estima energia por inferência com base em constantes de hardware típicas de um AVR:

```c
const float VCC      = 5.0;    // Tensão de alimentação (V)
const float I_ACTIVE = 0.020;  // Corrente ativa estimada (A)
const float POWER_W  = VCC * I_ACTIVE;
// Energia (J) = POWER_W × duração (s)
```
