import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Design da página
 
sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Projeto 2 - Previsão de Renda",
     page_icon=":?:",
     layout="wide",
)



# Carregar os dados
df = pd.read_csv("./input/previsao_de_renda.csv")

renda = df.dropna()

dados = pd.get_dummies(renda)

def pagina1():
    st.title('Calculo da Renda')
    st.write('Coloque suas informações cadastrais para a previsão de renda do banco.')

    barra_progresso = st.progress(0)

    # Simulação de carregamento
    for i in range(100):
        barra_progresso.progress(i + 1)
        time.sleep(0.02)

    # Dividir os dados em features (X) e target (y)
    X = dados.drop("renda", axis=1)
    y = dados["tempo_emprego"]

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento do modelo
    modelo = DecisionTreeRegressor()
    modelo.fit(X_train, y_train)

    # Função para fazer previsões para novos dados
    def prever_renda_novo_cliente(dados_novos):
        # Fazer a previsão de renda usando o modelo treinado
        previsao_renda = modelo.predict(dados_novos)
        return previsao_renda

    # Interface do usuário usando Streamlit
    st.title("Previsão de Renda")

    # Criar campos para entrada de dados do usuário
    sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
    posse_veiculo = st.radio("Possui Veículo?", ["Sim", "Não"])
    posse_imovel = st.radio("Possui Imóvel?", ["Sim", "Não"])
    tempo_emprego = st.number_input('Tempo empregado?')
    qtd_filhos = st.number_input('Quantidade de Filhos?')
    tipo_renda = st.radio("Tipo de Renda", ["Assalariado", "Empresário", "Pensionista", "Bolsista", "Servidor público"])
    estado_civil = st.radio("Estado Civil", ["Casado", "Solteiro", "União", "Separado", "viuvo"])
    educacao = st.radio("Nivel de Educação", ["Secundário", "Superior completo", "Superior incompleto", "Primário", "Pós graduação"])
    tipo_residencia = st.radio("Tipo de Residencia", ["Casa", "Com os pais", "Aluguel", "Studio"])
    idade = st.number_input('Idade?')
    qt_pessoas_residencia = st.number_input('Quantidade de Pessoas na Residencia?')

    # Criar um DataFrame com os dados inseridos pelo usuário
    dados_usuario = pd.DataFrame({
        "sexo_Feminino": [1 if sexo == "Feminino" else 0],
        "posse_veiculo_Sim": [1 if posse_veiculo == "Sim" else 0],
        "posse_imovel_Sim": [1 if posse_imovel == "Sim" else 0],
        "tempo_emprego": [tempo_emprego],
        "qtd_filhos": [qtd_filhos],
        "tipo_renda": [tipo_renda],
        "estado_civil": [estado_civil],
        "educacao": [educacao],
        "tipo_residencia": [tipo_residencia],
        "idade": [idade],
        "qt_pessoas_residencia": [qt_pessoas_residencia]
    })

    # Garantir que todas as features presentes nos dados de treinamento estejam presentes nos dados do usuário
    for coluna in X.columns:
        if coluna not in dados_usuario.columns:
            dados_usuario[coluna] = 0

    dados_usuario = dados_usuario[X.columns]

    if st.button('Calcular Renda'):
        # Fazer a previsão de renda para o novo cliente
        previsao = prever_renda_novo_cliente(dados_usuario)
        st.write("A previsão de renda para o novo cliente é:", previsao)
    






def pagina2():
    st.title('Gráficos')
    st.write('# Análise exploratória da previsão de renda')
    barra_progresso = st.progress(0)

    # Simulação de carregamento
    for i in range(100):
        barra_progresso.progress(i + 1)
        time.sleep(0.02)

    fig, ax = plt.subplots(8,1,figsize=(10,70))
    renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
    st.write('## Gráficos ao longo do tempo')
    sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
    ax[2].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
    ax[3].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
    ax[4].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
    ax[5].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
    ax[6].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
    ax[7].tick_params(axis='x', rotation=45)
    sns.despine()
    st.pyplot(plt)

    st.write('## Gráficos bivariada')
    fig, ax = plt.subplots(7,1,figsize=(10,50))
    sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
    sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
    sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
    sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
    sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
    sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
    sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
    sns.despine()
    st.pyplot(plt)    

# Barra Lateral 
    

def main():
    st.sidebar.title('Navegação')
    opcao = st.sidebar.radio('', ['Previsão de Renda', 'Gráficos'])

    if opcao == 'Previsão de Renda':
        pagina1()
    elif opcao == 'Gráficos':
        pagina2()


if __name__ == "__main__":
    main()

    
#Botão para as rede sociais

st.sidebar.markdown("<div style='position: fixed; bottom: 0; left: 0; padding: 10px;'><a href='https://www.linkedin.com/in/vinicius-paiotti-leduc/'><img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white'></a></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='position: fixed; bottom: 0; left: 120px; padding: 10px;'><a href='https://github.com/ViniciusPaiotti'><img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white'></a></div>", unsafe_allow_html=True)