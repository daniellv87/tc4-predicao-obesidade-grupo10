# 🩺 Sistema de Classificação de Níveis de Obesidade

## 📌 Sobre o Projeto
Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 4** da pós-graduação em Data Analytics. O objetivo é criar um modelo preditivo para auxiliar profissionais da saúde na identificação do grau de obesidade de pacientes com base em hábitos alimentares, estilo de vida e dados biométricos.

## 🚀 Tecnologias Utilizadas
* **Linguagem:** Python
* **Bibliotecas:** Pandas, Scikit-Learn, Seaborn, Joblib
* **Modelo:** Random Forest Classifier
* **Deploy:** Streamlit

## 📊 Principais Insights
* O modelo atingiu **96.93% de assertividade** (Accuracy).
* A substituição de Peso/Altura pela métrica de **IMC** otimizou a performance e a relevância clínica.
* Identificamos um provável viés de relato na variável de consumo de alimentos entre refeições, um ponto de atenção importante para diagnósticos clínicos.

## 📁 Estrutura do Repositório
* `app.py`: Código fonte da aplicação Streamlit.
* `modelo_obesidade_imc.pkl`: Modelo treinado exportado.
* `colunas_modelo_imc.pkl`: Ordem das variáveis esperadas pelo modelo.
* `requirements.txt`: Lista de dependências para o deploy.
* `Analise_Exploratoria.ipynb`: Notebook com todo o processo de ETL e treinamento.
