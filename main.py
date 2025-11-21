#region Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.inspection import permutation_importance
#endregion

#region Funções utilitárias
def print_titulo(titulo):
    print("\n------------------------------------------------------------------")
    print(titulo)
    print("------------------------------------------------------------------")
#endregion

#region Carregamento e visão inicial
print_titulo("Leitura do CSV e visão inicial")
df = pd.read_csv('./TMEDTREND_PUBLIC_250827.csv')

print("Colunas do dataset:")
print(df.columns)

df = df[df['Bene_Geo_Desc'] != 'National']
df = df.dropna(subset=['Pct_Telehealth'])
df = df.reset_index(drop=True)

print_titulo("Formato do dataset após filtros")
print(df.shape)

df['target'] = 0
for q in df['quarter'].unique():
    filtro = df['quarter'] == q
    limite = df.loc[filtro, 'Pct_Telehealth'].quantile(0.75)
    df.loc[filtro & (df['Pct_Telehealth'] >= limite), 'target'] = 1

print_titulo("Distribuição da variável alvo")
print(df['target'].value_counts())

print_titulo("Amostra dos dados tratados")
print(df.head())
#endregion

#region Exploração de Dados (EDA)
print_titulo("Análise exploratória - gráficos e resumos")

plt.style.use('default')

plt.figure(figsize=(8,4))
plt.hist(df['Pct_Telehealth'], bins=30, edgecolor='black')
plt.title('Distribuição de Pct_Telehealth')
plt.xlabel('Pct_Telehealth')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='Year', y='Pct_Telehealth')
plt.title('Pct_Telehealth por Year')
plt.show()

plt.figure(figsize=(5,4))
df['target'].value_counts().plot(kind='bar')
plt.title('Quantidade por classe target')
plt.xticks([0,1], ['Baixa 0', 'Alta 1'])
plt.xlabel('Classe')
plt.ylabel('Quantidade')
plt.show()

num_cols = ['Total_Bene_TH_Elig', 'Total_PartB_Enrl', 'Total_Bene_Telehealth', 'Pct_Telehealth']
corr = df[num_cols].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('Correlação entre variáveis numéricas')
plt.show()

print_titulo("Médias de Pct_Telehealth por perfis")
print("\nPor idade:")
print(df.groupby('Bene_Age_Desc')['Pct_Telehealth'].mean().sort_values(ascending=False).head(10))
print("\nPor sexo:")
print(df.groupby('Bene_Sex_Desc')['Pct_Telehealth'].mean().sort_values(ascending=False))
print("\nPor raça:")
print(df.groupby('Bene_Race_Desc')['Pct_Telehealth'].mean().sort_values(ascending=False))
print("\nPor rural ou urbano:")
print(df.groupby('Bene_RUCA_Desc')['Pct_Telehealth'].mean().sort_values(ascending=False))
#endregion

#region Pré-processamento para ML
print_titulo("Pré-processamento para Machine Learning")

cat_cols = ['Bene_Geo_Desc', 'quarter', 'Bene_Mdcd_Mdcr_Enrl_Stus', 'Bene_Race_Desc', 'Bene_Sex_Desc', 'Bene_Age_Desc', 'Bene_RUCA_Desc']
num_cols = ['Total_Bene_TH_Elig', 'Total_PartB_Enrl', 'Total_Bene_Telehealth', 'Year']

for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna('Unknown')

for c in num_cols:
    if c in df.columns:
        med = df[c].median()
        df[c] = df[c].fillna(med)

if 'Pct_Telehealth' in df.columns:
    df = df.drop(columns=['Pct_Telehealth'])

df_dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=True)

X = pd.concat([df[num_cols].reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
y = df['target'].astype(int).reset_index(drop=True)

print_titulo("Formato de X e y")
print("X:", X.shape)
print("y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print_titulo("Shapes após split e distribuição da classe")
print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("\nDistribuição y_train")
print(y_train.value_counts(normalize=True))
print("\nDistribuição y_test")
print(y_test.value_counts(normalize=True))

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = X_train_num
X_test_scaled[num_cols] = X_test_num

print_titulo("Exemplo de features e número final de features")
print(X_train_scaled.columns[:10])
print(X_train_scaled.shape[1])
#endregion

#region Treino e avaliação de modelos
print_titulo("Treino e avaliação de modelos")

def avaliar_modelo(nome, modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print_titulo("Resultado do modelo " + nome)
    print("Acurácia:", "{:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Precisão:", "{:.4f}".format(precision_score(y_test, y_pred)))
    print("Recall:", "{:.4f}".format(recall_score(y_test, y_pred)))
    print("F1:", "{:.4f}".format(f1_score(y_test, y_pred)))
    print("\nMatriz de confusão")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de classificação")
    print(classification_report(y_test, y_pred))

log_reg = LogisticRegression(max_iter=500)
avaliar_modelo("Logistic Regression", log_reg, X_train_scaled, X_test_scaled, y_train, y_test)

knn = KNeighborsClassifier(n_neighbors=5)
avaliar_modelo("KNN", knn, X_train_scaled, X_test_scaled, y_train, y_test)

svm = SVC(kernel='rbf', probability=True, random_state=42)
avaliar_modelo("SVM", svm, X_train_scaled, X_test_scaled, y_train, y_test)

print_titulo("Modelos treinados e avaliados com sucesso")
#endregion

#region Curva ROC e AUC comparativa
print_titulo("Curva ROC comparativa e AUC")

y_pred_proba_logreg = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
y_scores_svm = svm.decision_function(X_test_scaled)

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_proba_logreg)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_scores_svm)

auc_logreg = roc_auc_score(y_test, y_pred_proba_logreg)
auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
auc_svm = roc_auc_score(y_test, y_scores_svm)

print("AUC Logistic Regression", "{:.4f}".format(auc_logreg))
print("AUC KNN", "{:.4f}".format(auc_knn))
print("AUC SVM", "{:.4f}".format(auc_svm))

plt.figure()
plt.plot(fpr_logreg, tpr_logreg, label="LogReg AUC {:.3f}".format(auc_logreg))
plt.plot(fpr_knn, tpr_knn, label="KNN AUC {:.3f}".format(auc_knn))
plt.plot(fpr_svm, tpr_svm, label="SVM AUC {:.3f}".format(auc_svm))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("Falso Positivo FPR")
plt.ylabel("Verdadeiro Positivo TPR")
plt.title("Curva ROC comparativa")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
#endregion

#region Permutation importance
print_titulo("Permutation importance no conjunto de teste")

res = permutation_importance(svm, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)

imp_mean = res.importances_mean
imp_std = res.importances_std
feat_names = list(X.columns)
idx_desc = np.argsort(imp_mean)[::-1]

top_n = 15
top_idx = idx_desc[:top_n]
top_feats = [feat_names[i] for i in top_idx]
top_imp = imp_mean[top_idx]
top_std = imp_std[top_idx]

print_titulo("Top features por importance")
for i, (f, val) in enumerate(zip(top_feats, top_imp), 1):
    print(str(i) + ".", f, "->", "{:.6f}".format(val))

plt.figure(figsize=(8, max(4, 0.25*len(top_feats))))
y_pos = np.arange(len(top_feats))
plt.barh(y_pos, top_imp[::-1], xerr=top_std[::-1], align='center')
plt.yticks(y_pos, top_feats[::-1])
plt.xlabel("Decrease in score mean")
plt.title("Permutation importance top features")
plt.tight_layout()
plt.show()

print_titulo("Pipeline finalizado. Verifique as saídas para montar o relatório")
#endregion
