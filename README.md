# Challenge

## Visão geral

Este repositório contém a análise e o código para responder à pergunta central do escopo: **quais características demográficas e regionais estão associadas a maior probabilidade de alta adoção de telemedicina** (unidades = estado × trimestre × recorte demográfico). O procedimento transforma `Pct_Telehealth` em uma classe binária (top 25% por trimestre → alta adoção) e usa modelos de classificação para identificar sinais preditivos.

## Estrutura do `main.py`

O script acompanha o pipeline linear e comentado com blocos (regions). Principais etapas:

1. Imports
2. Funções utilitárias (apenas para uma impressão organizada do títulos)
3. Carregamento e visão inicial dos dados
4. Exploração de dados (EDA): histogramas, boxplots, heatmap e médias por perfil
5. Pré-processamento: preenchimento, one-hot encoding, padronização das colunas numéricas e split estratificado (80/20) com `random_state=42`
6. Treino e avaliação de modelos: Logistic Regression, KNN (k=5) e SVM (RBF) — avaliação no conjunto de teste
7. Curva ROC comparativa e cálculo de AUC
8. Permutation importance sobre o conjunto de teste para explicabilidade

## Requisitos e instalação

Instale as dependências mínimas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Como rodar

1. Coloque `TMEDTREND_PUBLIC_250827.csv` na mesma pasta do `main.py` ou atualize o caminho no script.
2. No terminal, execute:

```bash
python main.py
```

O script imprimirá mensagens estruturadas no console e exibirá gráficos interativos (ou janelas de figura, dependendo do ambiente). Saídas importantes aparecem ordenadas com separadores para facilitar coleta.

## Saídas principais (resumo dos resultados obtidos)

Com os dados e código final executado, os resultados principais foram:

* Observações após filtros: **25 171** linhas
* Distribuição da classe alvo: **Baixa (0) = 18 875**, **Alta (1) = 6 296** (≈ 75% / 25%)
* Modelos treinados e métricas no conjunto de teste:

| Modelo              |   Acurácia | Precisão (classe 1) | Recall (classe 1) | F1 (classe 1) |        AUC |
| ------------------- | ---------: | ------------------: | ----------------: | ------------: | ---------: |
| Logistic Regression |     0.8967 |              0.8350 |            0.7315 |        0.7798 |     0.9539 |
| KNN (k=5)           |     0.8618 |              0.7313 |            0.7069 |        0.7189 |     0.9093 |
| **SVM (RBF)**       | **0.9358** |          **0.9000** |        **0.8364** |    **0.8670** | **0.9831** |

* Permutation importance (top 15) indicou `Total_Bene_Telehealth`, `Total_Bene_TH_Elig`, `Year` e `Total_PartB_Enrl` como as variáveis com maior impacto médio no score do SVM.

## Interpretação resumida com foco na pergunta orientadora

Com base nas evidências:

* **Escala de uso** (número absoluto de usuários de telemedicina) e **tamanho da população elegível** são os sinais mais associados à alta adoção.
* **Fatores temporais** mostram tendência de crescimento ao longo dos anos/trimestres e ajudam a identificar recortes mais maduros.
* **Características demográficas** como predominância urbana e maior presença de faixas etárias mais jovens aumentam a probabilidade de alta adoção, mas têm efeito secundário em relação à escala.
* **SVM** foi o modelo que melhor sintetizou esses sinais no conjunto testado e por isso é indicado para priorização de recortes.

Estas conclusões respondem diretamente à pergunta do escopo e são baseadas em métricas e métodos de explicabilidade aplicados ao conjunto de teste.

## Créditos

Trabalho realizado pela equipe:

- Caio Freitas - RM553190
- Caio Hideki - RM553630
- Jorge Booz - RM552700
- Mateus Tibão - RM553267
- Lana Andrade - RM552596
