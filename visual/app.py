import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Detecção de Ataques Cibernéticos - TCC",
    page_icon="🛡️",
    layout="wide"
)

sns.set_style("darkgrid")
plt.rcParams['figure.dpi'] = 120

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_METRICAS = os.path.join(BASE, "Sistema - TCC", "resultados", "metricas")
PASTA_FIGURAS = os.path.join(BASE, "Sistema - TCC", "figuras")
PASTA_ZERO_DAY = os.path.join(BASE, "Sistema - TCC", "dados", "dados_processados", "zero_day")

ALGORITMOS = {
    "Random Forest": {
        "arquivo": "metricas_random_forest.csv",
        "cor": "#3498db",
        "supervisionado": True,
        "tem_roc": False
    },
    "XGBoost": {
        "arquivo": "metricas_xgboost.csv",
        "cor": "#e67e22",
        "supervisionado": True,
        "tem_roc": False
    },
    "Isolation Forest": {
        "arquivo": "metricas_isolation_forest.csv",
        "cor": "#27ae60",
        "supervisionado": False,
        "tem_roc": True
    },
    "Mean Shift": {
        "arquivo": "metricas_mean_shift.csv",
        "cor": "#9b59b6",
        "supervisionado": False,
        "tem_roc": True
    }
}

@st.cache_data
def carregar_metricas(algoritmo):
    info = ALGORITMOS[algoritmo]
    caminho = os.path.join(PASTA_METRICAS, info["arquivo"])
    if not os.path.exists(caminho):
        st.error(f"Arquivo não encontrado: {caminho}")
        return None
    df = pd.read_csv(caminho)
    return df

def plot_metricas_barras(df, algoritmo):
    metricas_base = ["Acurácia (%)", "Precisão (%)", "Recall (%)", "F1-Score (%)"]
    if ALGORITMOS[algoritmo]["tem_roc"]:
        metricas = metricas_base + ["ROC-AUC (%)"]
    else:
        metricas = metricas_base
    versoes = df["Versão"].tolist()
    cores = ["#3498db", "#e67e22", "#27ae60"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metricas))
    largura = 0.25

    for i, versao in enumerate(versoes):
        valores = [df.loc[df["Versão"] == versao, m].values[0] for m in metricas]
        bars = ax.bar(x + i * largura, valores, largura, label=versao,
                      color=cores[i], edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Métrica", fontsize=11)
    ax.set_ylabel("Valor (%)", fontsize=11)
    ax.set_title(f"Métricas de Classificação — {algoritmo}", fontsize=13, fontweight="bold")
    ax.set_xticks(x + largura)
    ax.set_xticklabels(metricas, rotation=15, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    return fig

def plot_tempos_barras(df, algoritmo):
    versoes = df["Versão"].tolist()
    tempos_treino = df["Tempo Treino (s)"].values
    tempos_pred = df["Tempo Pred. (s)"].values

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(versoes))
    largura = 0.35

    bars1 = ax.bar(x - largura / 2, tempos_treino, largura,
                   label="Treino", color="steelblue", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + largura / 2, tempos_pred, largura,
                   label="Predição", color="coral", edgecolor="black", linewidth=0.8)

    for bar, v in zip(bars1, tempos_treino):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}s", ha="center", fontsize=9, fontweight="bold")
    for bar, v in zip(bars2, tempos_pred):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.4f}s", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Versão", fontsize=11)
    ax.set_ylabel("Tempo (segundos)", fontsize=11)
    ax.set_title(f"Tempos de Execução — {algoritmo}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(versoes, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_matriz_confusao(cm, classes, titulo, acuracia):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                ax=ax, cbar=True, square=True,
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    ax.set_xlabel("Predito", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.set_title(f"{titulo}\n(Acurácia: {acuracia:.2f}%)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return fig

def plot_comparativo_global(dfs):
    metricas_base = ["Acurácia (%)", "Precisão (%)", "Recall (%)", "F1-Score (%)"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    x = np.arange(len(metricas_base))
    largura = 0.2

    for i, (alg, df) in enumerate(dfs.items()):
        melhor = df.loc[df["F1-Score (%)"].idxmax()]
        valores = [melhor[m] for m in metricas_base]
        ax.bar(x + i * largura, valores, largura, label=alg,
               color=ALGORITMOS[alg]["cor"], edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Métrica", fontsize=11)
    ax.set_ylabel("Valor (%)", fontsize=11)
    ax.set_title("Comparação Global — Melhor Versão de Cada Algoritmo", fontsize=13, fontweight="bold")
    ax.set_xticks(x + largura * 1.5)
    ax.set_xticklabels(metricas_base, rotation=15, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 105])

    ax = axes[1]
    nomes = []
    f1s = []
    cores = []
    versoes_melhores = {}
    for alg, df in dfs.items():
        melhor = df.loc[df["F1-Score (%)"].idxmax()]
        nomes.append(alg)
        f1s.append(melhor["F1-Score (%)"])
        cores.append(ALGORITMOS[alg]["cor"])
        versoes_melhores[alg] = melhor["Versão"]

    bars = ax.barh(nomes[::-1], f1s[::-1], color=cores[::-1], edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, f1s[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("F1-Score (%)", fontsize=11)
    ax.set_title("Ranking por F1-Score (Melhor Versão)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim([0, 110])

    legenda = [f"{alg}: {ver}" for alg, ver in versoes_melhores.items()]
    ax.text(0.95, 0.05, "\n".join(legenda), transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    return fig

# ─── Funções para Ataques Desconhecidos (Zero-Day) ────────────────────────────

@st.cache_data
def carregar_zero_day(nome_arquivo):
    caminho = os.path.join(PASTA_ZERO_DAY, nome_arquivo)
    if not os.path.exists(caminho):
        st.error(f"Arquivo não encontrado: {caminho}")
        return None
    return pd.read_csv(caminho)

CORES_ZERO = {"Random Forest": "#3498db", "XGBoost": "#e67e22",
              "Isolation Forest": "#27ae60", "Mean Shift": "#9b59b6"}
CORES_VERSAO = {"ORIGINAL": "#2c3e50", "PCA": "#e74c3c", "LDA": "#16a085"}

def plot_zero_day_barras(df, metrica="F1-Score"):
    fig, ax = plt.subplots(figsize=(12, 5))
    algos = df["Algoritmo_Completo"].unique()
    x = np.arange(len(algos))
    largura = 0.25
    for i, versao in enumerate(["ORIGINAL", "PCA", "LDA"]):
        vals = []
        for alg in algos:
            row = df[(df["Algoritmo_Completo"] == alg) & (df["Versão"] == versao)]
            if not row.empty:
                vals.append(row[metrica].values[0] * 100)
            else:
                vals.append(0)
        bars = ax.bar(x + i * largura, vals, largura, label=versao,
                      color=CORES_VERSAO[versao], edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Algoritmo", fontsize=11)
    ax.set_ylabel(f"{metrica} (%)", fontsize=11)
    ax.set_title(f"{metrica} — Ataques Desconhecidos (Heartbleed + Infiltration)", fontsize=13, fontweight="bold")
    ax.set_xticks(x + largura)
    ax.set_xticklabels(algos, fontsize=9)
    ax.legend(title="Versão", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 110])
    plt.tight_layout()
    return fig

def plot_zero_day_deteccao(df_det):
    fig, ax = plt.subplots(figsize=(12, 5))
    modelos = df_det["Modelo"].tolist()
    heartbleed = df_det["Heartbleed (%)"].values
    infiltration = df_det["Infiltration (%)"].values
    x = np.arange(len(modelos))
    largura = 0.35
    bars1 = ax.bar(x - largura / 2, heartbleed, largura, label="Heartbleed",
                   color="#e74c3c", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + largura / 2, infiltration, largura, label="Infiltration",
                   color="#8e44ad", edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars1, heartbleed):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", fontsize=7, fontweight="bold")
    for bar, v in zip(bars2, infiltration):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Modelo", fontsize=11)
    ax.set_ylabel("Taxa de Detecção (%)", fontsize=11)
    ax.set_title("Taxa de Detecção por Tipo de Ataque Desconhecido", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 115])
    plt.tight_layout()
    return fig

def plot_zero_day_fpr_fnr(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    modelos = df["Modelo"].tolist()
    fpr = df["FPR"].values * 100
    fnr = df["FN"].values / (df["TP"].values + df["FN"].values) * 100
    x = np.arange(len(modelos))
    largura = 0.35
    bars1 = ax.bar(x - largura / 2, fpr, largura, label="FPR (Falso Positivo)",
                   color="#e67e22", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + largura / 2, fnr, largura, label="FNR (Falso Negativo)",
                   color="#c0392b", edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars1, fpr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=7, fontweight="bold")
    for bar, v in zip(bars2, fnr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Modelo", fontsize=11)
    ax.set_ylabel("Taxa (%)", fontsize=11)
    ax.set_title("Taxa de Falsos Positivos (FPR) e Falsos Negativos (FNR)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    return fig

def plot_zero_day_tempos(df_tempos):
    fig, ax = plt.subplots(figsize=(12, 4))
    algos = df_tempos["Algoritmo"].unique()
    x = np.arange(len(algos))
    largura = 0.25
    for i, versao in enumerate(["ORIGINAL", "PCA", "LDA"]):
        vals = []
        for alg in algos:
            row = df_tempos[(df_tempos["Algoritmo"] == alg) & (df_tempos["Versão"] == versao)]
            vals.append(row["Tempo (s)"].values[0] if not row.empty else 0)
        bars = ax.bar(x + i * largura, vals, largura, label=versao,
                      color=CORES_VERSAO[versao], edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Algoritmo", fontsize=11)
    ax.set_ylabel("Tempo (s)", fontsize=11)
    ax.set_title("Tempo de Inferência — Ataques Desconhecidos", fontsize=13, fontweight="bold")
    ax.set_xticks(x + largura)
    ax.set_xticklabels(algos, fontsize=9)
    ax.legend(title="Versão", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_zero_day_matrizes_confusao(df):
    ordem = [
        ("Random Forest", "ORIGINAL"), ("Random Forest", "PCA"), ("Random Forest", "LDA"),
        ("XGBoost", "ORIGINAL"), ("XGBoost", "PCA"), ("XGBoost", "LDA"),
        ("Isolation Forest", "ORIGINAL"), ("Isolation Forest", "PCA"), ("Isolation Forest", "LDA"),
        ("Mean Shift", "ORIGINAL"), ("Mean Shift", "PCA"), ("Mean Shift", "LDA"),
    ]
    cores_cmap = {"Random Forest": "Blues", "XGBoost": "Greens",
                  "Isolation Forest": "Purples", "Mean Shift": "Oranges"}

    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    fig.suptitle("Matrizes de Confusão — 12 Modelos em Zero-Day\n(0 = BENIGNO, 1 = ATAQUE)",
                 fontsize=14, fontweight="bold", y=1.00)

    for idx, (algoritmo, versao) in enumerate(ordem):
        row = df[(df["Algoritmo_Completo"] == algoritmo) & (df["Versão"] == versao)]
        if row.empty:
            continue
        r = row.iloc[0]
        tn, fp, fn, tp = int(r["TN"]), int(r["FP"]), int(r["FN"]), int(r["TP"])
        cm = np.array([[tn, fp], [fn, tp]])

        ax = axes[idx // 3, idx % 3]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cores_cmap[algoritmo],
                    xticklabels=["BENIGNO", "ATAQUE"],
                    yticklabels=["BENIGNO", "ATAQUE"],
                    cbar=False, ax=ax, linewidths=1, linecolor="white")
        ax.set_title(f"{algoritmo} ({versao})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predito", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)

    plt.tight_layout()
    return fig

st.title("🛡️ Detecção de Ataques Cibernéticos em Tráfego de Rede")
st.markdown("""
**Aplicação de algoritmos de Machine Learning para detecção de ataques cibernéticos em tráfego de rede**

Este dashboard apresenta os resultados da comparação entre **4 algoritmos** aplicados sobre um dataset com **~2 milhões de registros** de tráfego de rede, contendo ataques **DDoS, DoS Hulk, PortScan** e tráfego **BENIGN**.

Foram aplicadas técnicas de redução de dimensionalidade (**PCA** e **LDA**) para avaliar o desempenho dos modelos com diferentes quantidades de atributos.

Na seção **Ataques Desconhecidos**, são apresentados os resultados da aplicação dos mesmos modelos treinados sobre **ataques nunca vistos** (Heartbleed e Infiltration), simulando um cenário de **zero-day attack**.
""")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Por Algoritmo", "🏆 Comparação Global", "🆕 Ataques Desconhecidos", "ℹ️ Sobre o Projeto"])

with tab1:
    col_alg, col_ver = st.columns([1, 3])
    with col_alg:
        algoritmo = st.selectbox(
            "Selecione o algoritmo",
            list(ALGORITMOS.keys()),
            index=0
        )

    df = carregar_metricas(algoritmo)

    if df is not None:
        st.subheader(f"📈 Resultados — {algoritmo}")

        col_metrica, col_detalhe = st.columns([1, 1])
        with col_metrica:
            st.markdown("### Tabela de Métricas")
            metricas_exibir = [c for c in df.columns if c != "N Clusters"]
            st.dataframe(
                df[metricas_exibir].style
                .format({
                    "Acurácia (%)": "{:.2f}",
                    "Precisão (%)": "{:.2f}",
                    "Recall (%)": "{:.2f}",
                    "F1-Score (%)": "{:.2f}",
                    "ROC-AUC (%)": "{:.2f}",
                    "Tempo Treino (s)": "{:.4f}",
                    "Tempo Pred. (s)": "{:.6f}"
                })
                .background_gradient(subset=["Acurácia (%)", "Precisão (%)", "Recall (%)", "F1-Score (%)"], cmap="Blues"),
                use_container_width=True
            )

        with col_detalhe:
            st.markdown("### Melhor Versão")
            melhor_idx = df["F1-Score (%)"].idxmax()
            melhor = df.loc[melhor_idx]
            st.success(f"**{melhor['Versão']}** — F1-Score: **{melhor['F1-Score (%)']:.2f}%**")
            st.info(f"Acurácia: {melhor['Acurácia (%)']:.2f}%  |  Precisão: {melhor['Precisão (%)']:.2f}%  |  Recall: {melhor['Recall (%)']:.2f}%")

            if "ROC-AUC (%)" in melhor and not pd.isna(melhor["ROC-AUC (%)"]):
                st.info(f"ROC-AUC: {melhor['ROC-AUC (%)']:.2f}%")

            if "N Clusters" in df.columns:
                n_clusters = df.loc[melhor_idx, "N Clusters"]
                st.info(f"Nº de Clusters: {int(n_clusters)}")

            mais_rapido_treino = df.loc[df["Tempo Treino (s)"].idxmin(), "Versão"]
            mais_rapido_pred = df.loc[df["Tempo Pred. (s)"].idxmin(), "Versão"]
            st.caption(f"⚡ Treino mais rápido: **{mais_rapido_treino}**")
            st.caption(f"⚡ Predição mais rápida: **{mais_rapido_pred}**")

        st.divider()
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            fig_metrics = plot_metricas_barras(df, algoritmo)
            st.pyplot(fig_metrics)

        with col_graf2:
            fig_tempos = plot_tempos_barras(df, algoritmo)
            st.pyplot(fig_tempos)

        if "N Clusters" in df.columns:
            st.divider()
            st.markdown("### Número de Clusters por Versão")
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(df["Versão"], df["N Clusters"], color=["#3498db", "#e67e22", "#27ae60"],
                          edgecolor="black", linewidth=0.8)
            for bar, v in zip(bars, df["N Clusters"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{int(v)}", ha="center", fontsize=12, fontweight="bold")
            ax.set_xlabel("Versão", fontsize=11)
            ax.set_ylabel("Número de Clusters", fontsize=11)
            ax.set_title(f"Nº de Clusters por Versão — {algoritmo}", fontsize=13, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.subheader("🏆 Comparação entre Todos os Algoritmos")

    dfs = {}
    for alg in ALGORITMOS:
        d = carregar_metricas(alg)
        if d is not None:
            dfs[alg] = d

    if dfs:
        fig_global = plot_comparativo_global(dfs)
        st.pyplot(fig_global)

        st.divider()
        st.markdown("### Tabela Comparativa — Melhor Versão de Cada Algoritmo")

        dados_melhores = []
        for alg, df in dfs.items():
            melhor = df.loc[df["F1-Score (%)"].idxmax()].to_dict()
            melhor["Algoritmo"] = alg
            dados_melhores.append(melhor)

        df_melhores = pd.DataFrame(dados_melhores).reset_index(drop=True)
        cols_ordem = ["Algoritmo", "Versão", "Acurácia (%)", "Precisão (%)", "Recall (%)", "F1-Score (%)",
                      "Tempo Treino (s)", "Tempo Pred. (s)"]
        if "ROC-AUC (%)" in df_melhores.columns:
            cols_ordem.insert(6, "ROC-AUC (%)")
        cols_exibir = [c for c in cols_ordem if c in df_melhores.columns]

        st.dataframe(
            df_melhores[cols_exibir].style
            .format({
                "Acurácia (%)": "{:.2f}",
                "Precisão (%)": "{:.2f}",
                "Recall (%)": "{:.2f}",
                "F1-Score (%)": "{:.2f}",
                "ROC-AUC (%)": "{:.2f}",
                "Tempo Treino (s)": "{:.4f}",
                "Tempo Pred. (s)": "{:.6f}"
            })
            .background_gradient(subset=["Acurácia (%)", "Precisão (%)", "Recall (%)", "F1-Score (%)"], cmap="viridis"),
            use_container_width=True
        )

        st.divider()
        st.markdown("### 📋 Resumo da Comparação")

        melhor_f1 = max(dfs, key=lambda a: dfs[a].loc[dfs[a]["F1-Score (%)"].idxmax(), "F1-Score (%)"])
        melhor_val = dfs[melhor_f1].loc[dfs[melhor_f1]["F1-Score (%)"].idxmax()]
        st.success(f"**Melhor algoritmo geral:** **{melhor_f1}** (versão **{melhor_val['Versão']}**) "
                   f"com F1-Score de **{melhor_val['F1-Score (%)']:.2f}%** e Acurácia de **{melhor_val['Acurácia (%)']:.2f}%**")

        ranking = sorted(
            dfs.keys(),
            key=lambda a: dfs[a].loc[dfs[a]["F1-Score (%)"].idxmax(), "F1-Score (%)"],
            reverse=True
        )
        for i, alg in enumerate(ranking, 1):
            melhor_linha = dfs[alg].loc[dfs[alg]["F1-Score (%)"].idxmax()]
            medalha = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}º")
            st.write(f"{medalha} **{alg}** — F1: {melhor_linha['F1-Score (%)']:.2f}% "
                     f"(versão {melhor_linha['Versão']}, Acc: {melhor_linha['Acurácia (%)']:.2f}%)")

        st.divider()
        st.markdown("### ⚡ Eficiência (Tempo)")
        mais_rapido_treino_global = min(dfs.keys(),
                                        key=lambda a: dfs[a]["Tempo Treino (s)"].min())
        mais_rapido_pred_global = min(dfs.keys(),
                                       key=lambda a: dfs[a]["Tempo Pred. (s)"].min())
        st.write(f"🏎️ **Treinamento mais rápido:** {mais_rapido_treino_global}")
        st.write(f"🏎️ **Predição mais rápida:** {mais_rapido_pred_global}")

with tab3:
    st.subheader("🆕 Ataques Desconhecidos (Zero-Day)")

    st.markdown("""
    ### 🧪 Cenário
    Os modelos foram treinados apenas com as classes **BENIGN, DDoS, DoS Hulk e PortScan**.  
    Agora, os mesmos modelos (sem qualquer ajuste adicional) são aplicados sobre **2 ataques nunca vistos**:

    - **Heartbleed** (11 registros) — vulnerabilidade OpenSSL
    - **Infiltration** (36 registros) — varredura interna com exploração

    > **Objetivo:** Avaliar a capacidade de generalização dos modelos diante de **ataques zero-day** completamente desconhecidos.
    """)

    df_zero = carregar_zero_day("metricas_zero_day.csv")
    df_rank = carregar_zero_day("ranking_zero_day.csv")
    df_det = carregar_zero_day("deteccao_por_classe.csv")
    df_tempos = carregar_zero_day("tempos_inferencia.csv")

    if df_zero is not None:
        st.divider()
        st.markdown("### 📊 Métricas de Classificação Binária (BENIGN vs ATAQUE)")

        metricas_display = df_zero.copy()
        for col in ["Acurácia", "Precisão", "Recall", "F1-Score"]:
            metricas_display[col] = metricas_display[col] * 100

        st.dataframe(
            metricas_display[["Algoritmo_Completo", "Versão", "Acurácia", "Precisão", "Recall", "F1-Score", "FPR"]]
            .style
            .format({
                "Acurácia": "{:.2f}",
                "Precisão": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-Score": "{:.2f}",
                "FPR": "{:.2f}"
            })
            .background_gradient(subset=["F1-Score"], cmap="Greens"),
            use_container_width=True
        )

        st.divider()
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            fig_f1 = plot_zero_day_barras(df_zero, "F1-Score")
            st.pyplot(fig_f1)
        with col_graf2:
            fig_recall = plot_zero_day_barras(df_zero, "Recall")
            st.pyplot(fig_recall)

        st.divider()
        col_graf3, col_graf4 = st.columns(2)
        with col_graf3:
            fig_det = plot_zero_day_deteccao(df_det)
            st.pyplot(fig_det)
        with col_graf4:
            fig_fpr = plot_zero_day_fpr_fnr(df_zero)
            st.pyplot(fig_fpr)

        st.divider()
        st.markdown("### 🔢 Matrizes de Confusão — 12 Modelos")
        fig_cm = plot_zero_day_matrizes_confusao(df_zero)
        st.pyplot(fig_cm)

        st.divider()
        st.markdown("### 🏆 Ranking — Top 5 Modelos")

        top5 = df_rank.head(5).copy()
        top5["F1-Score"] = top5["F1-Score"] * 100
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            medalha = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}º")
            st.success(
                f"{medalha} **{row['Algoritmo_Completo']}** ({row['Versão']}) — "
                f"F1: **{row['F1-Score']:.2f}%** | "
                f"Acurácia: {row['Acurácia']*100:.2f}% | "
                f"Recall: {row['Recall']*100:.2f}% | "
                f"Precisão: {row['Precisão']*100:.2f}% | "
                f"FPR: {row['FPR']*100:.1f}%"
            )

        st.divider()
        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.markdown("### ❤️ Detecção de Heartbleed")
            hb = df_det[["Modelo", "Heartbleed (det)", "Heartbleed (%)"]].copy()
            hb.columns = ["Modelo", "Detecção", "%"]
            hb["%"] = hb["%"] / 100
            st.dataframe(
                hb.style.format({"%": "{:.1%}"})
                .background_gradient(subset=["%"], cmap="Reds"),
                use_container_width=True
            )
        with col_det2:
            st.markdown("### 🔍 Detecção de Infiltration")
            inf = df_det[["Modelo", "Infiltration (det)", "Infiltration (%)"]].copy()
            inf.columns = ["Modelo", "Detecção", "%"]
            inf["%"] = inf["%"] / 100
            st.dataframe(
                inf.style.format({"%": "{:.1%}"})
                .background_gradient(subset=["%"], cmap="Purples"),
                use_container_width=True
            )

        st.divider()
        st.markdown("### ⏱️ Tempo de Inferência")
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            fig_tempos = plot_zero_day_tempos(df_tempos)
            st.pyplot(fig_tempos)
        with col_t2:
            tempo_total = df_tempos["Tempo (s)"].sum()
            tempo_medio = df_tempos["Tempo (s)"].mean()
            mais_rapido = df_tempos.loc[df_tempos["Tempo (s)"].idxmin()]
            mais_lento = df_tempos.loc[df_tempos["Tempo (s)"].idxmax()]
            st.metric("⏱️ Tempo Total (12 modelos)", f"{tempo_total:.4f}s")
            st.metric("⚡ Tempo Médio por Modelo", f"{tempo_medio:.4f}s")
            st.success(f"🏎️ **Mais rápido:** {mais_rapido['Algoritmo']} ({mais_rapido['Versão']}) — {mais_rapido['Tempo (s)']:.4f}s")
            st.warning(f"🐢 **Mais lento:** {mais_lento['Algoritmo']} ({mais_lento['Versão']}) — {mais_lento['Tempo (s)']:.4f}s")

        st.divider()
        st.markdown("### 📋 Resumo dos Resultados")

        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown("#### Algoritmos Supervisionados")
            sup = df_zero[df_zero["Algoritmo_Completo"].isin(["Random Forest", "XGBoost"])]
            st.metric("Média Recall (Supervisionados)", f"{sup['Recall'].mean()*100:.2f}%")
            st.metric("Média F1 (Supervisionados)", f"{sup['F1-Score'].mean()*100:.2f}%")
            st.caption("Os supervisionados tiveram **dificuldade** em detectar ataques desconhecidos, "
                       "pois aprenderam apenas os padrões das classes vistas no treinamento.")

        with col_sum2:
            st.markdown("#### Algoritmos Não Supervisionados")
            unsup = df_zero[df_zero["Algoritmo_Completo"].isin(["Isolation Forest", "Mean Shift"])]
            st.metric("Média Recall (Não Supervisionados)", f"{unsup['Recall'].mean()*100:.2f}%")
            st.metric("Média F1 (Não Supervisionados)", f"{unsup['F1-Score'].mean()*100:.2f}%")
            st.caption("Os não supervisionados **generalizaram melhor**, detectando a maioria dos ataques "
                       "por se tratar de anomalias em relação ao tráfego BENIGN.")

        st.info(
            "💡 **Conclusão:** Modelos não supervisionados (Isolation Forest e Mean Shift) são mais eficazes "
            "para detectar ataques zero-day, com destaque para **Mean Shift (PCA)** que atingiu "
            f"**{df_rank.iloc[0]['F1-Score']*100:.2f}%** de F1-Score. "
            "Já os supervisionados (Random Forest e XGBoost) falham em identificar ataques não vistos, "
            "pois seu aprendizado é restrito às classes do treinamento."
        )

with tab4:
    st.subheader("ℹ️ Sobre o Projeto")

    st.markdown("""
    ### 📌 Objetivo
    Comparar o desempenho de diferentes algoritmos de **Machine Learning** para detectar ataques cibernéticos em tráfego de rede.

    ### 📂 Dataset
    - **Fonte:** CIC-IDS2017 (UNB)
    - **Registros:** ~2 milhões
    - **Classes:** BENIGN, DDoS, DoS Hulk, PortScan
    - **Atributos originais:** 78 features de rede

    ### 🔧 Pré-processamento
    - Limpeza e tratamento de valores ausentes
    - Remoção de outliers
    - Amostragem estratificada (7.000 treino + 3.000 teste)

    ### 📉 Redução de Dimensionalidade
    - **PCA (15 atributos):** Análise de Componentes Principais
    - **LDA (3 atributos):** Análise Discriminante Linear

    ### 🤖 Algoritmos Avaliados
    | Algoritmo | Tipo | Característica |
    |---|---|---|
    | **Random Forest** | Supervisionado | Ensemble de árvores de decisão |
    | **XGBoost** | Supervisionado | Gradient boosting otimizado |
    | **Isolation Forest** | Não supervisionado | Detecção de anomalias |
    | **Mean Shift** | Não supervisionado | Clustering baseado em densidade |

    ### 📊 Métricas Utilizadas
    | Métrica | Descrição |
    |---|---|
    | **Acurácia** | Proporção total de acertos |
    | **Precisão** | Proporção de verdadeiros positivos entre os positivos previstos |
    | **Recall** | Proporção de verdadeiros positivos identificados corretamente |
    | **F1-Score** | Média harmônica entre precisão e recall |
    | **ROC-AUC** | Área sob a curva ROC (para modelos não supervisionados) |

    ### 🆕 Teste com Ataques Desconhecidos (Zero-Day)
    Após a avaliação inicial, os mesmos modelos foram submetidos a um cenário **zero-day** com **2 ataques nunca vistos** durante o treinamento:

    - **Heartbleed** (11 registros) — vulnerabilidade OpenSSL
    - **Infiltration** (36 registros) — varredura interna
    - **BENIGN** (100 registros) — tráfego normal não utilizado no treino

    **Principais descobertas:**
    - Modelos **não supervisionados** (Isolation Forest e Mean Shift) detectaram **~87%** dos ataques desconhecidos
    - Modelos **supervisionados** (Random Forest e XGBoost) detectaram apenas **~21%** em média
    - **Mean Shift com PCA** foi o melhor modelo geral, com **80,4%** de F1-Score e **14%** de FPR
    """)

st.divider()
st.caption("TCC — Aplicação de algoritmos de Machine Learning para detecção de ataques cibernéticos em tráfego de rede")