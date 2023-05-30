import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import *
import json
import pandas as pd

# Inicializar o objeto FastAPI
app = FastAPI()

# Definir uma classe de modelo para os dados do POST
class DataItem(BaseModel):
    empresa: int
    orcamento_situacao: str
    estoque_disponivel:int
    orcamento_qtd:int
    orcamento_pr_unitario:int
    orcamento_pr_percentualdesconto:int
    orcamento_pr_desconto:int
    orcamento_pr_desconto_total:int
    orcamento_pr_total_produto:int
    orcamento_pr_total:int
    orcamento_valor_total:int
    orcamento_pr_percentualmargem:int
    orcamento_pr_margem:int
    orcamento_pr_custo:int


@app.post("/")
def predicte(data: DataItem):
    # Lógica para criar os dados recebidos
    # Substitua este exemplo com sua própria lógica para criar os dados
    dado = {"empresa": data.empresa,
                    "orcamento_situacao": data.orcamento_situacao,
                    "estoque_disponivel":data.estoque_disponivel,
                    "orcamento_qtd":data.orcamento_qtd,
                    "orcamento_pr_unitario":data.orcamento_pr_unitario,
                    "orcamento_pr_percentualdesconto":data.orcamento_pr_percentualdesconto,
                    "orcamento_pr_desconto":data.orcamento_pr_desconto,
                    "orcamento_pr_desconto_total":data.orcamento_pr_desconto_total,
                    "orcamento_pr_total_produto":data.orcamento_pr_total_produto,
                    "orcamento_pr_total":data.orcamento_pr_total,
                    "orcamento_valor_total":data.orcamento_valor_total,
                    "orcamento_pr_percentualmargem":data.orcamento_pr_percentualmargem,
                    "orcamento_pr_margem":data.orcamento_pr_margem,
                    "orcamento_pr_custo":data.orcamento_pr_custo
                    }
    # Convert JSON string to Python dictionary
    #data_dict = json.loads(dado)

    # Convert dictionary to DataFrame
    dado = pd.DataFrame(dado, index=[0])

    best = load_model('./modelos/model-08')

    pred = predict_model(best,data=dado)
    score = float(pred.prediction_score)*1
    label = pred.prediction_label.values[0]

    if score<0.6:
        prob="baixa:"+str(score)
    elif score>=0.6 and score<0.8:
        prob="média:"+str(score)
    else: 
        prob="alta:"+str(score)

    if label=='Aberto':
        result = 'Cliente provavelment vai fechar negócio'
    else:
        result = 'Cliente provavelmente vai fechar negócio'

    return {'Resultado':result,
        'Probabilidade':prob
        }


def main():
    # Definir o título do aplicativo
    st.title("Minha API com Streamlit")

    # Exibir informações sobre a API
    st.write("API está em execução. Visualize a documentação em:")
    st.write("http://localhost:8000/docs")

# Executar a função principal
if __name__ == "__main__":
    main()
