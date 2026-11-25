# RunCast · Half Marathon

**Aplikacja online:** _(link pojawi się po wdrożeniu na DigitalOcean App Platform)_

Aplikacja Streamlit, w której użytkownik przedstawia się w polu tekstowym, a LLM (OpenAI) wyciąga z tekstu
wiek, płeć i czas na **5 km** (albo 10 km). Brakujące dane aplikacja wypisuje, a wywołania LLM trafiają do Langfuse.
Na tej podstawie podaje:

- **prognozę z przedziałem** — najbardziej prawdopodobny czas na mecie i zakres, w który na danych testowych trafiło ~80–84% biegaczy,
- **plan tempa** — międzyczasy na 5/10/15/20 km i wykres profilu tempa,
- **miejsce w stawce** — percentyl w grupie (płeć + dekada wieku) na tle rozkładu prawdziwych wyników,
- **kalkulator celu** — jaki czas na 5/10 km jest potrzebny, by złamać np. 1:45 (odwrócenie modelu).

## Realizacja zadania

| Wymaganie | Gdzie |
|---|---|
| Dane w Digital Ocean Spaces | `pipeline.ipynb`, sekcja 1 — wysyłka CSV do bucketu |
| Notebook-pipeline czyta dane ze Spaces | `pipeline.ipynb`, sekcja 2 (`boto3`) |
| Czyszczenie danych | `pipeline.ipynb`, sekcja 3 |
| Trening, metryki, feature selection | `pipeline.ipynb`, sekcje 4–5 (MAE, RMSE, R², porównanie zestawów cech, permutation importance) |
| Nowa wersja modelu lokalnie i w Spaces | `pipeline.ipynb`, sekcja 6 (`model/` + `runcast/models/` w buckecie) |
| Aplikacja Streamlit | `app.py` |
| Wdrożenie na App Platform | `.do/app.yaml` (`doctl apps create --spec .do/app.yaml`), `runtime.txt` |
| Pole tekstowe + informacja o brakujących danych | `app.py` — funkcje `brakujace`, `sprawdz` |
| LLM (OpenAI) → słownik JSON | `app.py` — funkcja `wyluskaj_dane` (`gpt-4o-mini`, `response_format=json_object`) |
| Langfuse — metryki skuteczności LLM | `app.py` — `@observe`, `langfuse.openai`, score `kompletnosc_danych` (jaka część danych została wyłuskana) |

## Model

Dane: Półmaraton Wrocławski 2023–2024, ~17,9 tys. finiszerów z kompletnymi międzyczasami.
`HistGradientBoostingRegressor` z regresją kwantylową (5%, 50% i 95%) i ograniczeniem monotoniczności,
osobno dla wejścia 5 km i 10 km, plus modele medianowe międzyczasów. Walidacja: trening 2023, test 2024.
Średni błąd prognozy z czasu na 5 km: **5,45 min** (wzór Riegela: 7,77 min).
Feature selection: czas na 5 km niesie prawie całą informację; wiek i płeć zostają (nie pogarszają modelu,
są wymagane jako wejście). Szczegóły w `pipeline.ipynb`, metryki w `model/metryki.json` i w zakładce „O modelu”.

## Uruchomienie lokalne

```bash
pip install -r requirements.txt
cp .env.example .env     # i wpisz klucze OpenAI, Langfuse i Spaces
# pipeline.ipynb — trening: dane z DO Spaces -> czyszczenie -> feature selection -> model lokalnie i w Spaces
# (train.py to starsza wersja tego samego treningu jako skrypt)
streamlit run app.py
```
