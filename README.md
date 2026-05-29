# Detecção de Ataques Cibernéticos em Tráfego de Rede

Comparação de algoritmos de **Machine Learning** para detecção de ataques cibernéticos em tráfego de rede, utilizando o dataset **CIC-IDS2017 (UNB)**.

## Algoritmos Avaliados

| Algoritmo | Tipo |
|---|---|
| Random Forest | Supervisionado |
| XGBoost | Supervisionado |
| Isolation Forest | Não supervisionado |
| Mean Shift | Não supervisionado |

## Estrutura do Projeto

```
Sistema - TCC/
├── dados/
│   ├── dados_nao_processados/  # Dados brutos CIC-IDS2017 (não versionados)
│   └── dados_processados/      # Dados após pré-processamento
│       ├── treino_teste/       # Conjuntos de treino e teste
│       └── zero_day/           # Resultados dos testes zero-day
├── modelos/                     # Modelos treinados (.pkl)
├── notebooks/                   # Pipeline completo em Jupyter Notebooks
├── resultados/
│   ├── metricas/                # Métricas de avaliação
│   └── graficos/                # Gráficos gerados
└── figuras/                     # Curvas ROC
visual/
└── app.py                       # Dashboard Streamlit
```

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o pipeline (opcional)

Execute os notebooks em ordem numérica:

1. `01_carregamento.ipynb`
2. `02_analise_exploratoria.ipynb`
3. `03_1_preprocessamento.ipynb`
4. `03_5_amostragem.ipynb`
5. `04_reducao_dimensionalidadev2.ipynb`
6. `05_divisao_treino_teste.ipynb`
7. `06_1_random_forest.ipynb` a `06_4_mean_shift.ipynb`
8. `07_1_ataques_desconhecidos.ipynb` a `07_4_ataques_desconhecidos.ipynb`

### 3. Executar o Dashboard

```bash
streamlit run visual/app.py
```

## Dataset

- **Fonte:** [CIC-IDS2017 (UNB)](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Registros:** ~2 milhões
- **Classes:** BENIGN, DDoS, DoS Hulk, PortScan
- **Atributos originais:** 78 features de rede

## Teste Zero-Day

Os modelos foram testados contra ataques nunca vistos durante o treinamento:
- **Heartbleed** (11 registros)
- **Infiltration** (36 registros)
