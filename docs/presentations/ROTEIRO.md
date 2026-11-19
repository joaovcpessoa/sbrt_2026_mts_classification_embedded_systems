### Título

Vou apresentar o trabalho "Classificação de Séries Temporais Multivariadas em Sistemas Embarcados com Recursos Limitados", feito junto com o professor e pesquisador Rafael Chaves e o professor e pesquisador Marcello de Campos. Nosso foco foi explorar o funcionamento da quantização pós-treinamento de redes neurais aplicadas em um dispositivo embarcado.

### Motivação

Falhas em poços de petróleo têm um custo altíssimo e em três frentes: financeira, ambiental e humana. Hoje, o monitoramento tradicional envia os dados para centros de controle onshore. Isso cria latência e uma dependência de conectividade entre o evento e a decisão. No modelo atual, na nuvem, a decisão leva de segundos a minutos e exige conexão contínua e infraestrutura de transmissão. Nossa proposta é levar a decisão para o poço. Com um dispositivo ESP32 co-localizado diretamente na plataforma, processando localmente, com decisão em milissegundos.

### Proposta

Nosso pipeline tem quatro etapas. Primeiro, a extração de features: janela de Hann de 1000 amostras, sobreposição de 50% e rótulo por moda. Segundo, o treinamento de uma MLP 1000 → 32 → 16 → 3 com Adam, 100 épocas e divisão 80/20. Terceiro, a quantização pós-treinamento, com escala simétrica absmax, para INT8, INT16 e INT32, usando calibração e quarto, a implantação embarcada no ESP32, utilizando o simulador Wokwi, com aritmética puramente inteira e requantização por bit-shift. Vou detalhar cada etapa nos próximos slides.

### 3W Dataset

Usamos o dataset 3W, uma base pública desenvolvida pela Petrobras, com eventos reais, simulados e desenhados à mão. São 2.228 eventos amostrados a 1 Hz, com 27 variáveis em 42 poços, e três classes: operação normal (NO), instabilidade de fluxo (FI) e severe slugging (SS). NO é o perfil de produção padrão. FI são distúrbios de fluxo não periódicos, que podem evoluir para SS e SS é uma instabilidade intensa, que pode danificar equipamentos. O sinal que usamos é o T-TPT, a temperatura no topo do tubo de produção, sensível a variações no regime de fluxo.

### Metodologia: janela e rótulo

Primeiro, a janela. Usamos uma janela de Hann deslizante com N = 1000 amostras, o que dá cerca de 16,7 minutos a 1 Hz. O passo é N/2, ou seja, 500 amostras, então há 50% de sobreposição, o que evita cortar transições de falha. Janelas incompletas recebem zero-padding. Para o rótulo de cada janela, usamos a moda dos rótulos ponto a ponto dentro do intervalo, a Equação 1 do artigo. Na prática, se uma anomalia ocupa mais de 50% da janela, o segmento inteiro é classificado como anômalo, o que dá robustez a ruído perto das bordas.

### Metodologia: arquitetura da MLP

A rede é uma MLP pequena. A entrada tem 1000 valores, que são a janela do T-TPT. Depois vêm duas camadas ocultas, de 32 e de 16 neurônios, com ReLU e uma saída de logits, um por classe. No treinamento, usamos Adam com taxa de aprendizado 10⁻³, função de perda cross-entropy, batch de 32, 100 épocas e divisão 80/20.

### Metodologia: quantização pós-treinamento

A PTQ é aplicada aos pesos e às ativações depois que o modelo FP32 já foi treinado, sem fine-tuning, para INT8, INT16 e INT32. O fator de escala é o absmax simétrico, o máximo do valor absoluto do tensor, dividido por 2 elevado a b menos 1, menos 1, onde b é a largura de bits do tipo alvo, mais um pequeno épsilon. Para quantizar um elemento, dividimos pela escala, arredondamos e saturamos com o clip nos limites do tipo inteiro. Com isso, o alcance dinâmico do tensor é mapeado simetricamente para os limites do inteiro, o que minimiza o erro de quantização para distribuições aproximadamente simétricas em torno de zero, uma característica comum dos pesos de redes neurais.

### Metodologia: pesos, vieses e requantização

Os pesos usam a escala dos próprios pesos. Os vieses usam o produto da escala da entrada pela escala dos pesos, para poderem ser somados diretamente ao acumulador. A Tabela II mostra os tipos em C: em 8 bits, pesos em int8, vieses e acumulador em int32; em 16 bits, pesos em int16, vieses em int32 e acumulador em int64; em 32 bits, pesos em int32 e tanto vieses quanto acumulador em int64. Entre as camadas, como não usamos FPU, o fator de reescala é aproximado por um inteiro de 31 bits e um deslocamento de bits. Assim, toda a aritmética do ESP32 permanece inteira.

### Explicação extra — Como validamos no ESP32

*Sugestão de posição: logo depois do slide 8, antes dos resultados. Pode ser falada sobre o slide 8 ou virar um slide novo. As frases entre colchetes são opcionais.*

Antes dos resultados, um comentário rápido sobre como rodamos isso no ESP32. Nós não usamos uma placa física: usamos o Wokwi, um simulador online de eletrônica que executa o programa compilado como se fosse o ESP32. Ele é gratuito para uso pessoal, e roda no navegador.

O simulador executa o mesmo código que iria para a placa, e a nossa aritmética é toda inteira. Por isso, esperamos que as predições coincidam com as de uma placa real. Testamos 100 amostras balanceadas entre as três classes, em 10 lotes de 10, para cada precisão.

O que o simulador não faz é medir consumo, e o tempo dele é um tempo de simulação. Então, o tempo vem do cronômetro do próprio simulador, e a energia é uma estimativa: uma potência constante assumida, de 0,1 W, multiplicada pelo tempo. Por isso, as comparações entre INT8, INT16 e INT32 são consistentes entre si, mas os valores absolutos de tempo e energia devem ser lidos como indicativos. [A referência FP32 foi medida no computador, então a comparação com os modelos inteiros também é um indicativo, não um teste de bancada.] O próximo passo natural é medir em uma placa real.

### Resultados: métricas

A Tabela III normaliza as métricas pelo baseline FP32, que tem 95,0% de acurácia, 7,91 mJ por inferência e 127,5 kB de memória. O INT8 reduz a energia estimada em 55% e a memória para 32 kB, mas perde 6 pontos de acurácia, com 13% de divergência de predições. O INT16 mantém os 95,0% de acurácia, com divergência zero e SNR de 67 dB, gastando 42% menos energia estimada e metade da memória do FP32. O INT32 tem o mesmo resultado do INT16, mas com o dobro de memória e sem ganho nenhum. Por isso, o INT16 é o melhor custo-benefício.

### Resultados: matrizes de confusão

Nas matrizes, as linhas são a classe real e as colunas, a predita. A classe FI é praticamente perfeita: 1,00 nos três modelos quantizados e 0,96 no FP32. A principal fonte de confusão é SS sendo confundida com FI. No FP32, 11% das amostras SS vão para FI; no INT16 e no INT32, 12%. No INT8, esse número sobe para 30%, e é aí que está a perda de acurácia que vimos no slide anterior.

### Resultados: latência e energia

Os gráficos mostram a latência e a energia estimada por inferência. Os modelos INT16 e INT32 são mais lentos e gastam mais energia que o INT8, porque o ESP32 não tem aritmética de 64 bits nativa: o acumulador é emulado com várias instruções de 32 bits, e isso domina tempo e energia. Mas o ponto principal é outro: o verdadeiro gargalo não é a inferência. Encher a janela de 1000 amostras a 1 Hz leva cerca de 1000 segundos, então a latência de inferência, na casa de dezenas de milissegundos, é desprezível.

### Conclusões

Resumindo o INT16: 95% de acurácia, idêntica ao FP32, 42% menos energia estimada, 50% menos memória e 0% de divergência nas predições. Tiramos três conclusões. Primeira, o INT8 serve para orçamentos extremos: custa 6% de acurácia, mas dá a menor energia estimada, 55% a menos, e só 32 kB de memória, e essa perda pode ser mitigada com quantization-aware training. Segunda, o INT32 não se justifica frente ao INT16: mesma acurácia e divergência nula, com o dobro de memória. Terceira, o gargalo está em encher a janela, não na inferência.

### Agradecimento

Obrigado pela atenção. Os QR codes levam ao repositório do projeto no GitHub e ao site do nosso laboratório, e nosso contato está na tela. Agradecemos às instituições cujos logos aparecem aqui: INCT, CNPq, CAPES e UFRJ, além do nosso laboratório SMT. Ficamos à disposição para perguntas.

---

### Explicação extra sobre validação

Rodamos o código inteiro no Wokwi, um simulador gratuito do ESP32. As predições são confiáveis; tempo e energia são estimativas.**

O [Wokwi](https://wokwi.com/) é um simulador de eletrônica que roda no navegador. Em vez de comprar e ligar uma placa, você monta o circuito virtual e roda o programa nele. Ele simula alguns microcontroladores da família ESP32 e outras placas. Ele consegue executar projetos em Arduino/C++, MicroPython, Rust ou até um firmware já compilado. No nosso caso, o programa que roda lá é o sketch (.ino) gerado pelo `experiment.py`. Ele é compilado e executado no simulador e imprime no monitor serial a predição, o tempo e a energia de cada amostra.

Ele possui um plano gratuito (Community) com simulações ilimitadas, com a condição de que os projetos sejam públicos e estejam dentro de um limite de uso dos recursos. Os planos pagos servem para projetos privados, bibliotecas próprias, maior quantidade de recursos computacionais etc.

A questão é: É fiel? Depende do que você mede. Então tem alguma pegadinha? Várias.

O simulador executa o programa compilado e a nossa aritmética é toda inteira, que é determinística. Esperamos, portanto, as mesmas predições de uma placa real.

O teste no simulador usou 100 amostras (34 NO, 33 FI e 33 SS), em 10 lotes de 10, por conta da limitação de memória do hardaware. Cada amostra vale 1 ponto percentual de acurácia e cerca de 3 pontos dentro de uma classe. A diferença entre 95% e 89% foram 6 amostras. Como foram usadas as mesmas 100 amostras na CPU do computador para o FP32, fazemos essa equivalência "forçada"

O tempo com o relógio simulado e a documentação dele não promete precisão de tempo. Além disso, por padrão ele limita a CPU simulada a cerca de 8 MHz para rodar mais rápido, enquanto um ESP32 real trabalha a até 240 MHz, então os tempos absolutos tendem a ser bem maiores do que seriam na placa. As comparações entre INT8, INT16 e INT32 continuam fazendo sentido, porque as três rodam sob a mesma limitação. Na parte do consumo o cálculo é feito por equações de estimativa, já que ele não possui simuladores de sensores de tensão ou corrente.

1. A energia dos modelos INT não é medida, é calculada. O sketch assume uma potência fixa de 0,1 W (5 V × 20 mA) e multiplica pelo tempo de cada inferência. Confere com a Tabela III: 3,561 mJ ÷ 35,611 ms = 0,1 W, e o mesmo vale para INT16 e INT32. Ou seja, a "energia" dos INT é o tempo reescalado, e não um consumo medido; os −55% e −42% dependem dessa suposição. Vale notar também que o ESP32 opera a 3,3 V, então o valor é uma suposição.
2. A linha FP32 vem de outro ambiente. Pelo código, o FP32 é avaliado no computador, com o CodeCarbon medindo tempo e energia, e os números da tabela são compatíveis com isso (0,505 ms e 7,91 mJ dão cerca de 15,7 W, um valor típico de PC). Os INT vêm do simulador, a 0,1 W. Portanto, comparar tempo e energia entre FP32 e INT (inclusive os "0,45×" e "0,58×" da tabela) mistura duas plataformas e duas formas de medir. Já a comparação entre INT8, INT16 e INT32 é de mesmo tipo.

4. "Memória" é o tamanho dos parâmetros. Os 127,5 / 32,0 / 63,9 / 127,7 kB são o espaço de pesos e vieses, não o uso total de RAM ou flash do programa.
5. A rede tem 5 saídas no código, e os slides dizem 3. Os rótulos do 3W usados são 0, 3 e 4, e o código cria uma camada de saída de tamanho 5 (as contas de memória da tabela só fecham com 5). As classes são 3, mas a camada final é 1000 → 32 → 16 → 5. Vale ajustar os slides 3 e 6 ou ter essa explicação pronta.

- "Vocês testaram numa placa real?" Não, só o simulador.
- "A energia foi medida?" Não, foi estimada.
- "Por que o FP32 aparece tão mais rápido?" Porque a medição do FP32 foi feita em uma CPU bem avançada e as dos inteiros no simulador. Não são diretamente comparáveis.
- "Por que ninguém usa só INT8?" Porque ele perde 6 pontos de acurácia e 30% das amostras SS viram FI. Ele serve para orçamentos extremos, e a perda pode ser reduzida com quantization-aware training.