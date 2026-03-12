from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pda_logic import PDA, convert_pda_to_cfg
from regex_logic import regex_to_min_dfa_json
from turing_logic import simulate_turing, parse_transitions, build_graph_json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── PDA ──────────────────────────────────────────────────────────────────────

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
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ─── Regex ────────────────────────────────────────────────────────────────────

@app.get("/regex/to-automaton")
async def regex_automaton(exp: str):
    """
    Convierte una expresión regular al DFA minimizado.
    Retorna JSON con states, edges y alphabet listos para Flutter.
    """
    try:
        result = regex_to_min_dfa_json(exp)
        return result
    except Exception as e:
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ─── Turing ───────────────────────────────────────────────────────────────────

class TuringRequest(BaseModel):
    states: str
    initial: str
    accepts: str
    transitions: str
    cinta: str
    head_pos: int = 0
    max_steps: int = 1000

@app.post("/turing/simulate")
async def turing_simulate(data: TuringRequest):
    """
    Simula una Máquina de Turing paso a paso.
    Retorna los pasos de simulación y el resultado (ACCEPTED/REJECTED/TIMEOUT).
    """
    try:
        estados = [s.strip() for s in data.states.split(',') if s.strip()]
        aceptados = [s.strip() for s in data.accepts.split(',') if s.strip()]
        trans_dict = parse_transitions(data.transitions)

        resultado = simulate_turing(
            states=estados,
            transitions=trans_dict,
            initial_state=data.initial.strip(),
            accept_states=aceptados,
            tape_input=data.cinta,
            head_pos=data.head_pos,
            max_steps=data.max_steps,
        )
        return resultado  # {"steps": [...], "result": "ACCEPTED"|"REJECTED"|"TIMEOUT"}
    except Exception as e:
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/turing/graph")
async def turing_graph(data: TuringRequest):
    """
    Devuelve el grafo de la MT en formato AutomatonCanvas (states + edges).
    """
    try:
        estados = [s.strip() for s in data.states.split(',') if s.strip()]
        aceptados = [s.strip() for s in data.accepts.split(',') if s.strip()]
        trans_dict = parse_transitions(data.transitions)

        graph = build_graph_json(
            states=estados,
            transitions=trans_dict,
            initial=data.initial.strip(),
            accept_states=aceptados,
        )
        return graph  # {"states": [...], "edges": [...]}
    except Exception as e:
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)