# Reconhecedor facial — Inteligência Artificial

Projeto possui **experimentos** e **aplicação**.

## Instalação (Python 3.10–3.12 recomendado)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements-treino.txt
```

Apenas o reconhecedor com SFace: `pip install -r requirements.txt`.

## Parte 1

```bash
python -m experimentos.experimento --epocas 3
```

## Parte 2

```bash
python -m app.metric_learning cadastrar --nome Gabriel --quantidade 20 --consentimento
python -m app.metric_learning reconhecer --limiar 0.40
python -m app.metric_learning excluir --nome Gabriel
```

```text
testes/
  name1/foto1.jpg
  name2/foto2.jpg
```

```bash
python -m app.metric_learning avaliar --pasta testes --limiar 0.40
```

`--camera 1`.

## Teste

```bash
python -m unittest discover -s tests -v
python -m compileall -q app experimentos tests
```