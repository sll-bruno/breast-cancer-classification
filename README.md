# Breast Cancer Classification — Rede Neural do Zero com Interpretabilidade

MLP implementada **do zero em NumPy** para classificar tumores de mama como **benignos** ou **malignos** no dataset _Breast Cancer_ , com análise de interpretabilidade via **DeepSHAP** (também implementado do zero).

Forward, backpropagation, mini-batch SGD, ReLU, Softmax + Cross-Entropy e DeepSHAP são todos feitos manualmente. Uma versão equivalente em PyTorch existe apenas como baseline de comparação.

---

## Sumário

- [Estrutura do Repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Arquitetura da Rede](#arquitetura-da-rede)
- [Pipeline de Treinamento](#pipeline-de-treinamento)
- [Interpretabilidade (DeepSHAP)](#interpretabilidade-deepshap)

---

## Estrutura do Repositório

```
.
├── main.ipynb                 # Notebook principal: treino, avaliação e SHAP
├── data_exploration.ipynb     # Exploração do dataset
├── numpy_pytorch_comp.ipynb   # Comparação NumPy vs PyTorch
│
├── neural_network.py          # Classe NeuralNetwork (fit/predict/explain_instance)
├── dense_layer.py             # Camada densa (forward, backward, init xavier/he/random)
├── activation_function.py     # ReLU e SoftmaxCrossEntropy (+ rotinas DeepSHAP)
├── nn_pytorch.py              # Mesma arquitetura em PyTorch (baseline)
│
├── requirements.txt
└── README.md
```

---

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch seaborn   # extras: baseline PyTorch e beeswarm
```

Dependências principais: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `seaborn`.

---

## Como Executar

```bash
jupyter notebook main.ipynb
```

Ordem sugerida:

1. **`data_exploration.ipynb`** — distribuição das 30 features e do alvo.
2. **`main.ipynb`** — treino, avaliação e análise SHAP.
3. **`numpy_pytorch_comp.ipynb`** — comparação numérica entre as duas implementações.

---

## Arquitetura da Rede

MLP totalmente conectada em forma de funil:

```
Input (30 features)
   └─► Dense(32) ─ ReLU
        └─► Dense(16) ─ ReLU
             └─► Dense(8) ─ ReLU
                  └─► Dense(2)  ──► Softmax + Cross-Entropy
                                    [maligno, benigno]
```

- **Inicialização**: Xavier por padrão (também suporta He e random).
- **Bias**: inicializado em zero.
- **Ativações**: ReLU nas camadas ocultas; Softmax acoplada à Cross-Entropy na saída, com gradiente fechado `A − Y`.

---

## Pipeline de Treinamento

| Item          | Valor                                             |
| ------------- | ------------------------------------------------- |
| Split         | 80% treino / 20% validação (`np.random.seed(42)`) |
| Normalização  | z-score com média/desvio do **treino**            |
| Otimizador    | SGD com mini-batch                                |
| Batch size    | 16                                                |
| Learning rate | 1e-3                                              |
| Épocas        | 300                                               |
| Loss          | Cross-Entropy                                     |

Saídas automáticas:

- Curvas de **loss** e **acurácia** (treino × validação) por época.
- Marca de **early-stop** na época de menor loss de validação.

---

## Interpretabilidade (DeepSHAP)

DeepSHAP implementado do zero em `neural_network.py:34` (`explain_instance`) e nas rotinas `forward_baseline` / `backward_deepshap` de cada camada. A regra usada é a _rescale rule_ do DeepLIFT, base teórica do DeepSHAP.

**Funcionamento:**

1. Escolhe-se um **baseline** (vetor médio do treino — ≈ origem após normalização).
2. Faz-se forward da instância **e** do baseline, guardando ativações de ambos.
3. Propaga-se de volta um sinal one-hot na classe alvo, multiplicando por:
   - `Wᵀ` em camadas densas;
   - `Δy/Δx` (rescale rule) na ReLU.
4. Multiplica-se o "multiplier" final por `(x − baseline)` → contribuição SHAP por feature.

**Visualizações:**

- **Bar plot local** — top 10 features da instância (verde = a favor, vermelho = contra).
- **Bar plot global** — média de `|SHAP|` sobre 50 amostras de validação.
- **Beeswarm** — distribuição de SHAP por feature, colorida pelo valor normalizado da feature (azul = baixo, vermelho = alto).

---
