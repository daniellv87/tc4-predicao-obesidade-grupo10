import streamlit as st
import pandas as pd
import joblib

# Configuração da página
st.set_page_config(page_title="Preditor de Obesidade - Tech Challenge", layout="wide")

# Carregar o modelo e as colunas
modelo = joblib.load('modelo_obesidade_imc.pkl')
colunas = joblib.load('colunas_modelo_imc.pkl')

# Dicionários de tradução (Inverso do que fizemos no Encoding)
opcoes_sim_nao = {"Sim": 1, "Não": 0}
opcoes_genero = {"Masculino": 1, "Feminino": 0}
opcoes_frequencia = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}

st.title("🩺 Sistema de Apoio ao Diagnóstico: Obesidade")
st.markdown("Insira os dados do paciente abaixo para obter a classificação preditiva.")

# Criando a interface de entrada
col1, col2, col3 = st.columns(3)

with col1:
    genero = st.selectbox("Gênero", list(opcoes_genero.keys()))
    idade = st.number_input("Idade", min_value=1, max_value=120, value=25)
    altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    peso = st.number_input("Peso (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)

with col2:
    historico = st.selectbox("Histórico Familiar de Sobrepeso?", list(opcoes_sim_nao.keys()))
    favc = st.selectbox("Consome alimentos calóricos com frequência?", list(opcoes_sim_nao.keys()))
    h2o = st.slider("Consumo diário de água (Litros)", 1.0, 3.0, 2.0)
    calc = st.selectbox("Monitora as calorias que consome?", list(opcoes_sim_nao.keys()))

with col3:
    caec = st.selectbox("Consome alimentos entre as refeições?", list(opcoes_frequencia.keys()))
    faf = st.slider("Frequência de atividade física (Dias/Semana)", 0, 3, 1)
    transporte = st.selectbox("Utiliza transporte ativo (Caminhada/Bike)?", ["Sim", "Não"])

# Lógica de cálculo e predição
if st.button("📊 Realizar Diagnóstico"):
    # 1. Calcular o IMC
    imc = round(peso / (altura ** 2), 2)
    
    # 2. Montar o dicionário de entrada na ordem exata das colunas do modelo
    dados_paciente = {
        'genero': opcoes_genero[genero],
        'idade': idade,
        'historico_familiar': opcoes_sim_nao[historico],
        'frequencia_alimentos_caloricos': opcoes_sim_nao[favc],
        'frequencia_vegetais': 2, # Valor médio ou peça no formulário
        'numero_refeicoes': 3,    # Valor médio ou peça no formulário
        'alimentos_entre_refeicoes': opcoes_frequencia[caec],
        'fumante': 0,             # Valor padrão ou peça no formulário
        'consumo_agua': h2o,
        'monitora_calorias': opcoes_sim_nao[calc],
        'frequencia_atividade_fisica': faf,
        'tempo_de_tela': 1,       # Valor médio
        'consumo_alcool': 1,      # Valor médio
        'transporte_ativo': 1 if transporte == "Sim" else 0,
        'imc': imc
    }
    
    # Criar DataFrame e garantir a ordem das colunas
    df_entrada = pd.DataFrame([dados_paciente])[colunas]
    
    # Realizar Predição
    predicao = modelo.predict(df_entrada)[0]
    
    # Mapeamento da resposta
    categorias = {
        0: "Abaixo do Peso", 1: "Peso Normal", 2: "Sobrepeso Grau I",
        3: "Sobrepeso Grau II", 4: "Obesidade Grau I", 5: "Obesidade Grau II", 6: "Obesidade Grau III"
    }
    
    st.success(f"**Resultado do Diagnóstico:** {categorias[predicao]}")
    st.info(f"**IMC Calculado:** {imc}")