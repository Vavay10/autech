from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64

# Importamos las funciones lógicas de tus archivos
from AP import PDA, convert_pda_to_cfg
from expre import parse_regex, build_nfa, nfa_to_dfa, minimize_dfa, draw_automata

app = FastAPI()

# Permitir que Flutter Web acceda a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para recibir la definición de un PDA
class PDARequest(BaseModel):
    states: str
    input_alpha: str
    stack_alpha: str
    start_state: str
    start_symbol: str
    accept_states: str
    transitions: str

@app.post("/pda/to-cfg")
async def get_cfg(data: PDARequest):
    try:
        pda = PDA()
        pda.parse_from_ui(
            data.states, data.input_alpha, data.stack_alpha,
            data.start_state, data.start_symbol, data.accept_states,
            data.transitions
        )
        resultado = convert_pda_to_cfg(pda)
        return {"cfg": resultado}
    except Exception as e:
        print(f"Error detectado: {e}") # <--- ESTO te dirá el error exacto en la terminal
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/regex/to-image")
async def regex_image(exp: str):
    try:
        # Lógica de expre.py
        tokens = parse_regex(exp)
        nfa_s, nfa_t, nfa_i, nfa_f = build_nfa(tokens)
        dfa_s, dfa_t, dfa_i, dfa_a, alpha = nfa_to_dfa(nfa_s, nfa_t, nfa_i, nfa_f)
        min_s, min_t, min_i, min_a = minimize_dfa(dfa_s, dfa_t, dfa_i, dfa_a, alpha)
        
        # Obtenemos los bytes de la imagen
        img_bytes = draw_automata(min_s, min_t, min_i, min_a, alpha)
        # Lo convertimos a base64 para que Flutter lo lea fácil
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        return {"image_base64": encoded}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from turing3 import simular_maquina_turing_paso_a_paso

class TuringRequest(BaseModel):
    states: str
    initial: str
    accepts: str
    transitions: str
    cinta: str
    head_pos: int

@app.post("/turing/simulate")
async def simulate_turing(data: TuringRequest):
    try:
        # Convertir strings a estructuras de Python
        estados = [s.strip() for s in data.states.split(',') if s.strip()]
        aceptados = [s.strip() for s in data.accepts.split(',') if s.strip()]
        cinta_lista = list(data.cinta) if data.cinta else ['_']
        
        # Parsear transiciones del formato: q0,a->q1,X,R
        trans_dict = {}
        for line in data.transitions.strip().split('\n'):
            if '->' in line:
                orig, dest = line.split('->')
                state, char = orig.split(',')
                n_state, n_char, move = dest.split(',')
                if state.strip() not in trans_dict: trans_dict[state.strip()] = {}
                trans_dict[state.strip()][char.strip()] = (n_state.strip(), n_char.strip(), move.strip())

        pasos, resultado = simular_maquina_turing_paso_a_paso(
            estados, trans_dict, data.initial, aceptados, cinta_lista, data.head_pos
        )
        return {"pasos": pasos, "resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)