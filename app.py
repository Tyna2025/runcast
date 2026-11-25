"""RunCast · Half Marathon.

Użytkownik przedstawia się w polu tekstowym, LLM (OpenAI) wyciąga z tekstu
płeć, wiek i czas na 5 km do słownika, a model liczy prognozę czasu na mecie
z przedziałem, plan międzyczasów, miejsce w stawce i kalkulator celu.
Wywołania LLM trafiają do Langfuse. Model trenuje train.py -> model/tempomat.joblib.
"""
import json
import re
from datetime import date

import joblib
import openai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from langfuse import get_client, observe
from langfuse.openai import OpenAI

load_dotenv()

DYSTANS = 21.0975
ZAKRES = {5: (12 * 60, 60 * 60), 10: (26 * 60, 120 * 60)}  # dopuszczalny czas wejściowy [s]
WIEK = (16, 85)  # zakres wieku w danych treningowych
AKCENT, SZARY = "#C6F432", "#8B93A7"
MODEL_LLM = "gpt-4o-mini"
PROMPT = """Jesteś asystentem, który z wypowiedzi biegacza wyciąga dane do modelu.
Zwróć WYŁĄCZNIE obiekt JSON z kluczami:
- "plec": "K" dla kobiety, "M" dla mężczyzny, null gdy nie da się ustalić
  (możesz wnioskować z form gramatycznych, np. "biegałam" -> "K", "jestem biegaczem" -> "M"),
- "wiek": wiek w latach jako liczba całkowita, null gdy brak
  (jeśli podano rok urodzenia, policz wiek na rok {rok}),
- "czas_5km": czas na 5 km jako tekst "MM:SS" (np. "27 minut" -> "27:00", "26 i pół minuty" -> "26:30"), null gdy brak,
- "czas_10km": czas na 10 km jako tekst "MM:SS" albo "H:MM:SS", null gdy brak.
Formularz pyta o czas na 5 km, więc czas podany BEZ dystansu (np. "60 min") to czas na 5 km.
"czas_10km" wypełnij tylko wtedy, gdy użytkownik wprost wspomniał 10 km (albo "dychę").
Nie zgaduj wartości, których nie ma w tekście — wtedy wpisz null."""

st.set_page_config(page_title="RunCast · Half Marathon", page_icon="⏱️", layout="wide")
st.markdown(f"""<style>
.big {{ font-family: monospace; font-size: 4rem; font-weight: 700; color: {AKCENT}; line-height: 1; }}
.range {{ font-family: monospace; color: {SZARY}; margin-top: .4rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def wczytaj_model():
    return joblib.load("model/tempomat.joblib")


def parsuj_czas(txt):
    """'25:30' albo '1:02:10' -> sekundy, inaczej None."""
    m = re.fullmatch(r"\s*(?:(\d{1,2}):)?(\d{1,2}):([0-5]\d)\s*", txt or "")
    if not m:
        return None
    return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def fmt(sek, godziny=True):
    m, s = divmod(int(round(sek)), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if godziny else f"{m}:{s:02d}"


def cechy(wiek, kobieta, czas):
    return pd.DataFrame({"wiek": [wiek], "kobieta": [kobieta], "czas_wej": [czas]})


def prognoza(modele, km, wiek, kobieta, czas):
    """Zwraca (dół, mediana, góra) czasu mety oraz międzyczasy {km: sekundy}."""
    x = cechy(wiek, kobieta, czas)
    dol, p50, gora = (float(modele[km]["meta"][k].predict(x)[0]) for k in ("dol", "p50", "gora"))

    # Międzyczasy z modeli medianowych; ostatni odcinek (20 km -> meta) w tempie 15-20 km
    punkty = {km: float(czas)} | {s: float(m.predict(x)[0]) for s, m in modele[km]["splity"].items()}
    kms = sorted(punkty)
    odcinki = [max(punkty[b] - punkty[a], 1.0) for a, b in zip(kms, kms[1:])]
    odcinki.append(odcinki[-1] / 5 * (DYSTANS - 20))
    # Skalujemy odcinki, żeby ich suma zgadzała się z medianą mety
    skala = (p50 - czas) / sum(odcinki)
    splity, t = {km: float(czas)}, float(czas)
    for s, d in zip(kms[1:] + [DYSTANS], odcinki):
        t += d * skala
        splity[s] = t
    return min(dol, p50), p50, max(gora, p50), splity


def wymagany_czas(modele, km, wiek, kobieta, cel):
    """Bisekcja: model jest monotoniczny względem czasu wejściowego, więc da się go odwrócić."""
    model = modele[km]["meta"]["p50"]
    lo, hi = ZAKRES[km]
    if model.predict(cechy(wiek, kobieta, lo))[0] > cel:
        return None
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if model.predict(cechy(wiek, kobieta, mid))[0] <= cel else (lo, mid)
    return lo


@st.cache_resource
def klient_openai():
    return OpenAI()  # klucz z OPENAI_API_KEY (.env lokalnie, zmienne środowiskowe na App Platform)


def brakujace(dane):
    """Lista nazw danych, których LLM nie znalazł w tekście."""
    braki = []
    if dane.get("plec") not in ("K", "M"):
        braki.append("płeć")
    if dane.get("wiek") is None:
        braki.append("wiek")
    if not dane.get("czas_5km") and not dane.get("czas_10km"):
        braki.append("czas na 5 km")
    return braki


@observe(name="wyluskaj_dane")
def wyluskaj_dane(tekst):
    """LLM: tekst użytkownika -> słownik {plec, wiek, czas_5km, czas_10km}."""
    odp = klient_openai().chat.completions.create(
        model=MODEL_LLM,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT.format(rok=date.today().year)},
            {"role": "user", "content": tekst},
        ],
        name="ekstrakcja_danych",
    )
    surowe = json.loads(odp.choices[0].message.content)
    dane = {k: surowe.get(k) for k in ("plec", "wiek", "czas_5km", "czas_10km")}
    # Metryka skuteczności LLM w Langfuse: jaka część potrzebnych danych została wyłuskana
    braki = brakujace(dane)
    get_client().score_current_trace(
        name="kompletnosc_danych", value=round(1 - len(braki) / 3, 2),
        comment="brakuje: " + ", ".join(braki) if braki else "komplet",
    )
    return dane


def sprawdz(dane):
    """Słownik z LLM -> (wiek, kobieta, km, czas, braki, błędy)."""
    braki, bledy = brakujace(dane), []
    kobieta = int(dane.get("plec") == "K")
    wiek = dane.get("wiek")
    if wiek is not None:
        try:
            wiek = int(wiek)
            if not WIEK[0] <= wiek <= WIEK[1]:
                bledy.append(f"Wiek musi być od {WIEK[0]} do {WIEK[1]} lat (podano {wiek}).")
        except (TypeError, ValueError):
            bledy.append(f"Nie rozumiem wieku: {wiek}.")
    km, czas = None, None
    for k in (5, 10):
        txt = dane.get(f"czas_{k}km")
        if not txt:
            continue
        km, czas = k, parsuj_czas(str(txt))
        lo, hi = ZAKRES[k]
        if czas is None or not lo <= czas <= hi:
            bledy.append(f"Czas na {k} km musi być od {fmt(lo, False)} do {fmt(hi, False)} (podano {txt}).")
        break  # pierwszeństwo ma czas na 5 km
    return wiek, kobieta, km, czas, braki, bledy


def wykres(fig):
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ---------- dane wejściowe ----------
pakiet = wczytaj_model()
modele = pakiet["modele"]

st.title("⏱️ RunCast · Half Marathon")
st.caption("Model nauczony na wynikach Półmaratonu Wrocławskiego 2023–2024.")

with st.sidebar:
    st.header("Przedstaw się")
    with st.form("przedstaw_sie"):
        tekst = st.text_area(
            "Napisz kilka słów o sobie: płeć, wiek i czas na 5 km.",
            placeholder="Cześć, jestem Ania, mam 34 lata, 5 km biegam w 27 minut.",
            height=150,
        )
        wyslij = st.form_submit_button("Oblicz prognozę", type="primary", use_container_width=True)
    if wyslij:
        if not tekst.strip():
            st.session_state.pop("dane", None)
            st.warning("Napisz coś o sobie.")
        else:
            try:
                with st.spinner("Czytam Twoje dane…"):
                    st.session_state["dane"] = wyluskaj_dane(tekst)
            except openai.OpenAIError as e:
                st.session_state.pop("dane", None)
                st.error(f"Nie udało się połączyć z OpenAI: {e}")
    dane = st.session_state.get("dane")
    if dane:
        st.caption("Dane odczytane przez LLM")
        st.json(dane)

if not dane:
    st.info("👈 Przedstaw się w panelu po lewej, a policzę Twój czas na mecie.")
    st.stop()

wiek, kobieta, km, czas, braki, bledy = sprawdz(dane)
if braki or bledy:
    if braki:
        st.warning("Brakuje danych: **" + ", ".join(braki) + "**. Dopisz je w swoim opisie i kliknij „Oblicz prognozę”.")
    for b in bledy:
        st.error(b)
    st.stop()

dol, p50, gora, splity = prognoza(modele, km, wiek, kobieta, czas)
tempo = p50 / DYSTANS
r = pakiet["rozklad"]
dekada = min(max(wiek // 10 * 10, 20), 70)
grupa = r["meta"][(r["kobieta"] == kobieta) & (r["dekada"] == dekada)]
szybciej_niz = (grupa > p50).mean() * 100

prog, plan, cel_tab, o_modelu = st.tabs(["📍 Prognoza", "🗺️ Plan tempa", "🎯 Cel", "🔬 O modelu"])

# ---------- prognoza ----------
with prog:
    c1, c2 = st.columns([3, 2])
    c1.markdown("Najbardziej prawdopodobny wynik")
    c1.markdown(f'<div class="big">{fmt(p50)}</div>', unsafe_allow_html=True)
    # Pokrycie przedziału zmierzone na danych testowych (2024), a nie zakładane 90%
    pokrycie = pakiet["metryki"][f"{km}km"]["pokrycie_przedzialu"]
    c1.markdown(f'<div class="range">{pokrycie:.0%} szans: {fmt(dol)} – {fmt(gora)}</div>', unsafe_allow_html=True)
    c2.metric("Tempo", f"{fmt(tempo, False)} /km")
    # Grupa porównawcza: płeć + dekada wieku (w danych 16–29 lat to jedna grupa, 70+ też)
    wiek_grupy = "do 29 lat" if dekada == 20 else ("70+ lat" if dekada == 70 else f"{dekada}–{dekada + 9} lat")
    c2.metric(f"Szybciej niż ({'kobiety' if kobieta else 'mężczyźni'} {wiek_grupy})", f"{szybciej_niz:.0f}% biegaczy")

    st.subheader("Gdzie jesteś w stawce")
    fig = go.Figure(go.Histogram(x=grupa / 60, nbinsx=60, marker_color="#3A4560"))
    fig.add_vrect(x0=dol / 60, x1=gora / 60, fillcolor=AKCENT, opacity=.12, line_width=0)
    fig.add_vline(x=p50 / 60, line_color=AKCENT, line_width=3)
    fig.update_xaxes(title="czas na mecie [min]")
    fig.update_yaxes(title="liczba biegaczy")
    wykres(fig)

# ---------- plan tempa ----------
with plan:
    wiersze, poprz_t, poprz_km = [], 0.0, 0
    for k, t in sorted(splity.items()):
        tempo_odc = (t - poprz_t) / (k - poprz_km)
        wiersze.append({"km": k, "Punkt": "Meta" if k == DYSTANS else f"{k} km", "Międzyczas": fmt(t),
                        "Tempo odcinka": f"{fmt(tempo_odc, False)} /km", "tempo": tempo_odc})
        poprz_t, poprz_km = t, k
    df = pd.DataFrame(wiersze)

    c1, c2 = st.columns([2, 3])
    c1.dataframe(df[["Punkt", "Międzyczas", "Tempo odcinka"]], hide_index=True, use_container_width=True)
    fig = go.Figure(go.Scatter(x=df["km"], y=df["tempo"] / 60, mode="lines+markers",
                               line=dict(color=AKCENT, width=4, shape="hv")))
    fig.add_hline(y=tempo / 60, line_dash="dot", line_color=SZARY)
    fig.update_yaxes(autorange="reversed", title="tempo [min/km] (wyżej = szybciej)")
    fig.update_xaxes(title="km")
    with c2:
        wykres(fig)

# ---------- cel ----------
with cel_tab:
    cel = st.select_slider("Chcę złamać", ["1:30:00", "1:40:00", "1:45:00", "1:50:00", "2:00:00", "2:15:00"],
                           value="2:00:00")
    potrzeba = wymagany_czas(modele, km, wiek, kobieta, parsuj_czas(cel))
    if potrzeba is None:
        st.warning("Ten cel wykracza poza zakres danych, na których uczył się model.")
    else:
        c1, c2 = st.columns(2)
        c1.metric(f"Potrzebny czas na {km} km", fmt(potrzeba, False))
        c2.metric("Twój obecny", fmt(czas, False),
                  "masz to ✅" if czas <= potrzeba else f"brakuje {fmt(czas - potrzeba, False)}",
                  delta_color="normal" if czas <= potrzeba else "inverse")

# ---------- o modelu ----------
with o_modelu:
    st.markdown("""
Model to `HistGradientBoostingRegressor` z regresją kwantylową (5., 50. i 95. percentyl), osobno dla
wejścia 5 km i 10 km. Cechy: wiek, płeć i czas na wybranym dystansie. Osobne modele przewidują
międzyczasy na 10, 15 i 20 km. Walidacja: trening na edycji 2023, test na 2024.
""")
    met = pakiet["metryki"]
    st.dataframe(pd.DataFrame([
        {"Wejście": k, "Błąd modelu [min]": v["mae_min"], "Błąd wzoru Riegela [min]": v["mae_riegel_min"],
         "Ile prawdziwych wyników trafiło w zakres": f"{v['pokrycie_przedzialu']:.0%}"}
        for k, v in met.items() if isinstance(v, dict)]), hide_index=True)
    st.caption("Model uczył się na międzyczasach z samego półmaratonu, a Ty wpisujesz wynik z osobnego "
               "biegu, zwykle szybszy. Dlatego prognoza bywa lekko optymistyczna.")
