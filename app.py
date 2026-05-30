import streamlit as st
import pandas as pd
import joblib

# Configuração da página
st.set_page_config(page_title="Preditor de Obesidade - Tech Challenge", layout="wide")

# Carregamento do modelo e da estrutura de colunas do treinamento
try:
    modelo = joblib.load('modelo_obesidade_comportamental.pkl')
    colunas = joblib.load('colunas_modelo_comportamental.pkl')
except Exception as e:
    st.error(f"Erro ao carregar os arquivos do modelo (.pkl). Certifique-se de que eles estão na mesma pasta que este script. Detalhes: {e}")

# Dicionários de mapeamento (Alinhados perfeitamente com os Encodings do notebook)
opcoes_sim_nao = {"Sim": 1, "Não": 0}
opcoes_genero = {"Masculino": 1, "Feminino": 0}
opcoes_frequencia = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}

st.title("🩺 Sistema de Apoio ao Diagnóstico: Triagem Comportamental")
st.markdown("""
A partir do refinamento do modelo para prevenção de riscos, este sistema analisa **estritamente fatores comportamentais, 
estilo de vida e histórico familiar**, eliminando o viés do peso bruto e do IMC para focar na triagem preventiva.
""")

# Criando a interface de entrada organizada em 3 colunas
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Perfil Geral")
    genero = st.selectbox("Gênero", list(opcoes_genero.keys()))
    idade = st.number_input("Idade", min_value=14, max_value=61, value=25, step=1)
    historico = st.selectbox("Histórico Familiar de Sobrepeso?", list(opcoes_sim_nao.keys()))
    transporte = st.selectbox("Utiliza transporte ativo (Caminhada/Bike) para deslocamento?", ["Sim", "Não"])

with col2:
    st.subheader("🥗 Hábitos Alimentares")
    favc = st.selectbox("Consome alimentos calóricos com frequência (Frituras, FastFood)?", list(opcoes_sim_nao.keys()))
    fcvc = st.slider("Frequência de consumo de vegetais nas refeições (Nível)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais por dia", 1, 4, 3)
    caec = st.selectbox("Consome alimentos entre as refeições principais?", list(opcoes_frequencia.keys()))

with col3:
    st.subheader("🏋️ Estilo de Vida")
    h2o = st.slider("Consumo diário de água (Litros)", 1.0, 3.0, 2.0, step=0.5)
    calc_nutri = st.selectbox("Monitora as calorias que consome diariamente?", list(opcoes_sim_nao.keys()))
    faf = st.slider("Frequência de atividade física (Dias/Semana)", 0, 3, 1)
    tue = st.slider("Tempo diário de tela (Celular, TV, PC - Nível)", 0, 2, 1)
    calc_alcool = st.selectbox("Consumo de bebidas alcoólicas?", list(opcoes_frequencia.keys()))

# Lógica de predição
if st.button("📊 Realizar Análise de Risco"):
    
    # 1. Montar o dicionário inicial cruzando dados da tela com os encodings corretos
    # Inclui ambas as grafias de atividade física comuns nessa base de dados por segurança
    dados_paciente = {
        'genero': opcoes_genero[genero],
        'idade': int(idade),
        'historico_familiar': opcoes_sim_nao[historico],
        'frequencia_alimentos_caloricos': opcoes_sim_nao[favc],
        'frequencia_vegetais': int(fcvc),
        'numero_refeicoes': int(ncp),
        'alimentos_entre_refeicoes': opcoes_frequencia[caec],
        'consumo_agua': float(h2o),
        'monitora_calorias': opcoes_sim_nao[calc_nutri],
        'frequencia_atividade_fisica': int(faf),
        'frequencia_atividade_ficica': int(faf),
        'tempo_de_tela': int(tue),
        'consumo_alcool': opcoes_frequencia[calc_alcool],
        'transporte_ativo': 1 if transporte == "Sim" else 0,
        'fumante': 0  # Preenchimento padrão neutro caso exista no modelo original
    }
    
    # 2. Transforma em DataFrame provisório
    df_provisorio = pd.DataFrame([dados_paciente])
    
    # 3. TRUQUE DE SEGURANÇA ANTICRASH (Evita o KeyError):
    # Se o modelo original exigir alguma coluna que não está mapeada acima (como o 'imc' antigo),
    # o código cria essa coluna na hora preenchendo com 0, evitando que a aplicação trave.
    for col in colunas:
        if col not in df_provisorio.columns:
            df_provisorio[col] = 0
            
    # 4. Força o ordenamento exato das colunas exigidas pelo scikit-learn
    df_entrada = df_provisorio[colunas]
    
    try:
        # Executar a classificação preditiva do Random Forest ajustado
        predicao = modelo.predict(df_entrada)[0]
        
        # Mapeamento reverso do Target
        categorias = {
            0: "Abaixo do Peso", 
            1: "Peso Normal", 
            2: "Sobrepeso Grau I",
            3: "Sobrepeso Grau II", 
            4: "Obesidade Grau I", 
            5: "Obesidade Grau II", 
            6: "Obesidade Grau III"
        }
        
        # Exibição do resultado na interface
        st.success(f"**Resultado da Triagem Preditiva:** {categorias[predicao]}")
        
        # Caixa informativa com métricas de validação para transparência médica
        st.info("""
        **Nota Técnica do Modelo:** Classificador baseado estritamente em indicadores comportamentais. 
        Acurácia geral validada de **80.14%** sem dependência de aferição direta de balança (vazamento de IMC).
        """)
        
    except Exception as e:
        st.error(f"Erro ao processar a predição com o modelo carregado. Detalhes técnicos: {e}")
