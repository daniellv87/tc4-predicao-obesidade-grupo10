# 🩺 Sistema de Triagem e Análise de Risco Comportamental: Obesidade

## 📌 Sobre o Projeto
Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 4** da pós-graduação em Data Analytics. 

O objetivo é criar um modelo preditivo para auxiliar profissionais da saúde na identificação do grau de obesidade de pacientes com base em hábitos alimentares, estilo de vida e dados biométricos.

## 🚀 Tecnologias Utilizadas
* **Linguagem:** Python
* **Bibliotecas Analíticas:** Pandas, NumPy, Scikit-Learn, Joblib
* **Modelo Preditivo:** Random Forest Classifier (Otimizado contra Overfitting)
* **Deploy e Interface:** Streamlit Cloud

## 🧠 Abordagem Metodológica e Ajuste Fino
Para garantir a integridade clínica do modelo, o pipeline de Machine Learning passou por três grandes pilares de engenharia:
1. **Eliminação do IMC/Peso Bruto:** Forçou o algoritmo a aprender os reais gatilhos comportamentais ocultos em vez de apenas reproduzir uma fórmula matemática óbvia.
2. **Feature Selection (Filtro de Ruído):** Remoção de variáveis com correlação nula com o alvo (como a coluna `fumante`).
3. **Regularização (Hyperparameter Tuning):** Restrição da profundidade máxima (`max_depth=12`) e do tamanho das folhas (`min_samples_leaf=2`) do Random Forest para impedir que as árvores decorassem o ruído da base de treino.

## 📊 Performance e Validação
A performance deste classificador comportamental foi testada sob **Validação Cruzada (5-Fold Cross-Validation)** para provar sua capacidade de generalização:

* **Assertividade Geral nos Dados de Teste:** 80.14% (Superando a meta de 75% do desafio)
* **Acurácia Média na Validação Cruzada:** 77.79%

## 📁 Estrutura do Repositório
* `app.py`: Código-fonte da aplicação interativa em Streamlit.
* `modelo_obesidade_comportamental.pkl`: Modelo Random Forest otimizado e exportado.
* `colunas_modelo_comportamental.pkl`: Lista com a ordem exata das variáveis estruturadas que o modelo espera receber.
* `requirements.txt`: Arquivo de dependências limpo, contendo apenas os pacotes necessários para a produção no Streamlit Cloud.
* `Notebook_tc4_grupo10_semIMC.ipynb`: Notebook documentando todo o pipeline de ETL, Feature Encoding, Engenharia de Atributos, Descoberta do Viés de Relato e Modelagem.
