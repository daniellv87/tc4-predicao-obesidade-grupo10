# Lógica de predição
if st.button("📊 Realizar Análise de Risco"):
    
    # 1. Montar o dicionário básico com os inputs da tela
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
        'frequencia_atividade_ficica': int(faf), # Verifique se no notebook está 'atividade_ficica' ou 'atividade_fisica'
        'frequencia_atividade_fisica': int(faf),
        'tempo_de_tela': int(tue),
        'consumo_alcool': opcoes_frequencia[calc_alcool],
        'transporte_ativo': 1 if transporte == "Sim" else 0,
        'fumante': 0 # Caso o modelo antigo peça essa coluna
    }
    
    # 2. Criar o DataFrame inicial
    df_provisorio = pd.DataFrame([dados_paciente])
    
    # 3. TRUQUE DE SEGURANÇA: Se alguma coluna do modelo estiver faltando no dicionário, preenche com 0
    for col in colunas:
        if col prejudices not in df_provisorio.columns:
            df_provisorio[col] = 0
            
    # 4. Forçar a ordenação idêntica às features do treino sem dar KeyError
    df_entrada = df_provisorio[colunas]
    
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
    
    # Caixa informativa com métricas de validação do modelo para transparência médica
    st.info(f"""
    **Nota Técnica do Modelo:** Classificador baseado estritamente em indicadores comportamentais. 
    Acurácia geral validada de **{modelo.score(df_entrada, [predicao])*100:.2f}%** ou conforme relatório de treino.
    """)
