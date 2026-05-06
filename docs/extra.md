# Discussões importantes sobre o assunto

## Escolha do hardware

Escolher o hardware é importante pois a performance final será determinada pelo backend de quantização do hardware escolhido. O backend, nesse contexto, é o "motor" que realmente executa os cálculos matemáticos no hardware. Sem um backend específico, o processador tentaria rodar números inteiros usando as mesmas instruções de ponto flutuante, o que não traria ganho nenhum de velocidade. O backend traduz as operações da sua rede neural para instruções de baixo nível que o seu processador entende de forma otimizada.

As principais funções são:

- Aceleração de Hardware: Utiliza instruções especiais do processador (como AVX-512 em Intel ou NEON em chips ARM) para processar vários dados de uma vez (SIMD)
- Gerenciamento de Memória: Organiza como os pesos da rede são carregados no cache para evitar gargalos

## Quando um modelo é considerado "Integer Only"?

Um modelo é considerado 'integer-only' quando todas as suas operações computacionais (multiplicações, adições, ativações, etc.) são realizadas usando aritmética de inteiros de baixa precisão. Isso significa que não há conversões intermediárias para ponto flutuante durante a inferência. O fluxo de dados é puramente de inteiros do início ao fim do modelo.

O backend `fbgemm` do PyTorch (para CPUs) e o `TensorRT` da NVIDIA são capazes de executar modelos de forma totalmente quantizada para certas arquiteturas e operações.

A quantização híbrida (simulada/mixed-precision) é uma abordagem mais comum e flexível, especialmente durante o processo de desenvolvimento e calibração. Neste cenário, algumas operações são realizadas em inteiros, mas conversões intermediárias ainda podem ocorrer em ponto flutuante.

Embora isso possa parecer flexível, já que permite quantizar apenas as partes mais intensivas computacionalmente do modelo, mantendo outras partes em FP32 para preservar a acurácia ou simplificar a implementação, não funcionria em hardware embarcado que não possui suporte para FP32.

Com base na estrutura do exemplo, dá para perceber que é um exemplo de quantização 'híbrida'. O PyTorch adota a abordagem híbrida por padrão em PTQ por várias razões:

- Facilidade de Uso: Simplifica a integração com o ecossistema PyTorch existente, onde muitas operações e o treinamento são em FP32.

- Compatibilidade: Garante que o modelo possa interagir com outras partes do código que esperam entradas/saídas em FP32.

- Acurácia: Minimiza a perda de acurácia ao permitir que operações sensíveis (ou não otimizadas para inteiros) permaneçam em FP32 ou sejam dequantizadas temporariamente.

- Backends: A capacidade de executar um modelo de forma totalmente 'integer-only' depende do backend de quantização (e.g., fbgemm, qnnpack) e do hardware subjacente. O PyTorch tenta otimizar para o backend disponível, mas a estrutura com QuantStub/DeQuantStub é a representação padrão para a quantização simulada.

Para obter um modelo verdadeiramente 'integer-only' no PyTorch, você precisaria garantir que:

1. Todas as operações são suportadas: Cada operação no seu grafo computacional tem uma implementação otimizada para inteiros no backend de quantização escolhido.
2. Fusão de Operações: Operações como Conv + ReLU ou Linear + ReLU são fundidas em uma única operação quantizada para evitar dequantizações intermediárias.
3. Remoção de DeQuantStub: A DeQuantStub final seria removida ou posicionada apenas no ponto onde a saída precisa ser consumida por um sistema que espera FP32. Para inferência puramente em hardware de inteiros, a saída pode permanecer em INT8.

## Medição energética

Um diferencial desse código é o uso da biblioteca CodeCarbon. Enquanto o modelo treina, o EmissionsTracker fica "espiando" o hardware para calcular o tempo total de execução, a energia consumida (em Joules e Wh) e a estimativa de CO2 emitido para a atmosfera durante esse processamento.

Em cenários de Deep Learning para dispositivos móveis ou sistemas embarcados, medir a energia da inferência é quase tão importante quanto medir a acurácia. Uma coisa que eu poderia fazer é calcular apenas o custo do treino e separadamente o custo da inferência, assim eu treino uma relação de Custo de Operação vs. Custo de Criação.

No teste, estamos processando as amostras uma a uma (ou em pequenos lotes). Medir a energia aqui ajuda a entender a quanto custa para o processador classificar um único dígito. Se o modelo gasta muita energia para uma simples inferência de 10 mil amostras, ele pode ser inviável para rodar em um sensor que funciona a bateria, por exemplo.

Às vezes, um modelo "A" tem 99% de acurácia, mas gasta o dobro de energia que um modelo "B" com 98.5%. Ao medir a energia no teste, ganhamos dados para decidir: "Vale a pena gastar 50% mais bateria para ganhar 0.5% de precisão?"

### Como as equações de energia são calculadas nessa biblioteca?

Eu pessoalmente não gosto de utilizar uma "caixa preta". O ponto aqui é que a biblioteca não "chuta" esses valores, ela utiliza uma hierarquia de medição baseada no hardware disponível. 

É possível consultar os detalhes técnicos na documentação oficial ou no [repositório do GitHub](https://github.com/mlco2/codecarbon)  do projeto, mas aqui está o resumo das fórmulas principais:

A energia total ($E$) é a soma do consumo dos três componentes principais:

$$E_{total} = E_{cpu} + E_{gpu} + E_{ram}$$

Para cada componente, a energia consumida em um intervalo de tempo é calculada como:

$$E (kWh) = \frac{P (Watts) \times t (horas)}{1000}$$

As equações variam conforme o componente.

- GPU: Usa ferramentas como o `nvidia-smi` para ler diretamente o consumo de energia em tempo real reportado pela placa
- CPU: Intel: Usa a interface RAPL (Running Average Power Limit), que fornece leituras de energia extremamente precisas direto do processador
    - AMD: Usa o driver amd_energy (no Linux).
    - Fallback (Se não houver acesso ao hardware): Ele usa o TDP (Thermal Design Power) do modelo da CPU multiplicado pela carga de uso (cpu_load) atual.
- RAM: Como a maioria dos sistemas não reporta o consumo da RAM individualmente, aplica uma estimativa padrão (geralmente 0.375 W por GB de RAM instalada).
    
Para converter Energia em Emissões, a equação é:

$$CO_{2}e = E_{total} \times CI$$

Onde $CI$ é a Intensidade de Carbono da rede elétrica local. O CodeCarbon detecta sua localização via IP e busca em uma base de dados qual é a mistura energética do seu país (ex: se é baseada em hidrelétricas, como no Brasil, o $CI$ é baixo; se for carvão, é alto).

Se quiser fazer uma investigação mais profunda para futuras utilizações, dá para ver as equações exatas no código-fonte, basta procurar pelos módulos cpu.py, gpu.py e emissions.py dentro da pasta core no repositório oficial.

### Complicações de uso da biblioteca CodeCarbon

`[codecarbon WARNING] No CPU tracking mode found. Falling back on CPU load mode.`

Esse log é muito interessante porque revela exatamente como a biblioteca se comporta quando encontra "obstáculos" de permissão no Windows e como ele monitora o hardware moderno.

Como estou no Ruindows, a lib tentou acessar as interfaces de energia da AMD e não conseguiu. Por segurança, ela não tem acesso direto aos sensores de Watts do meu Ryzen 7 5700X3D.

Então a lib muda para o `cpu_load`, ou seja, ela pega o TDP do processador (105W para o 5700X3D) e multiplica pela porcentagem de uso da CPU naquele momento. Se o seu código está usando 10% da CPU, ele estima que você está consumindo ~10.5W.

Outra dúvida é que mesmo que o meu código PyTorch esteja rodando cálculos apenas na CPU,  a lib monitora o sistema como um todo, não apenas o processo do Python. Minha RTX 3060 está ligada, enviando imagem para o monitor e mantendo a interface do Windows rodando. Então ele chegou a calcular o consumo medido de apenas 3.88W. Isso é o gasto de energia da placa apenas por estar ligada (estado de repouso). A lib assume que, se você iniciou um experimento de IA, ele deve reportar toda a energia consumida pelo hardware de computação disponível, mesmo que a carga de trabalho específica não o esteja utilizando intensamente.

Para a RAM, ele identificou que tenho 32GB. Como não há um sensor padrão de energia para pentes de memória no Windows, ele usou o modelo estatístico:
`RAM power estimation model`, aplicou uma constante (normalmente 0.375W por GB) para chegar nos 20.0W que aparecem no log.

Então se quisermos calcular isso com maior fidelidade, é interessante testarmos em um sistema Linux, caso contrário, teremos que subtrair esses valores para chegar no mais próximo do real.

Usar um container Docker não adianta. Você só adiciona mais abstração e não tem acesso aos sensores.

### Como é feito o cálculo energético no microcontrolador?

Aqui para calcular a energia podemos usar os valores nominais. Qual o motivo? Sendo direto, não existe uma biblioteca C pura que consiga medir o consumo real de energia apenas via software. O Wokwi não possui sensores de corrente, como por exemplo INA219, que poderíamos utilizar para pegar os dados reais. O microcontrolador não tem um sensor de corrente interno. Ele sabe a que velocidade está operando, mas não sabe quanta eletricidade está puxando da fonte.

O que eu fiz foi, através da estimativa por ciclos de clock, como o Wokwi simula o tempo do processador com precisão, podemos usar o consumo médio de corrente do ATmega328P para calcular a energia. Um Arduino Uno consome cerca de 20mA operando a 5V. 

$$P = V \times I \implies 5V \times 0.02A = 0.1W$$

Multiplicamos essa potência fixa pelo tempo exato que o `mlp_predict` leva para rodar e *voilà*.

Como isso pode ser melhorado? Temos as seguintes opções:

1. Utilização de sensor

Eles raramente testam o dataset completo dentro do chip. Pelo que entendi de TinyML, o teste de 10.000 imagens serve para validar o modelo no CPU/GPU. No microcontrolador, o objetivo é a implantação real. Então eles podem utilizar um sensor de câmera acoplado ao Arduino capturaria uma imagem por vez. O modelo processa essa imagem única e descarta os dados para receber a próxima. Portanto, não há necessidade de guardar 10.000 imagens na memória; apenas o modelo (pesos) e o buffer da imagem atual precisam caber no chip.

2. Validação via Comunicação Serial

Utilização de um script de ponte:
- O hardware de treino, no caso a CPU, lê a imagem 1 do MNIST e envia os 784 bytes via cabo USB Serial para o Arduino
- O Arduino recebe os bytes, roda a predição e devolve apenas o número do resultado (ex: "7")
- A CPU compara com o rótulo real e passa para a imagem 2.

Isso permite testar milhões de amostras sem usar quase nada de Flash.

3. Uso de Hardware com Memória Externa

Para dispositivos que precisam operar isolados (offline) com muitos dados:
- Cartão SD: As imagens são lidas do SD, processadas e descartadas.
- Flash Externa (SPI Flash): Chips como o ESP32 ou o Raspberry Pi Pico permitem conectar memórias Flash externas de 4MB, 8MB ou até 16MB, onde você poderia armazenar uma parte muito maior do dataset.