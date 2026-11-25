"""RunCast — trening modeli: prognoza przedziałowa + profil tempa.

Dla każdego dystansu wejściowego (5 km albo 10 km) trenujemy:
  * trzy modele kwantylowe czasu na mecie (P5, P50, P95),
  * modele medianowe międzyczasów (10/15/20 km), z których powstaje plan tempa.

Walidacja: trening na 2023, test na 2024 (inna edycja, inna pogoda), potem
refit na obu latach. Wynik: model/tempomat.joblib i model/metryki.json.

Uruchomienie:  python train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA = Path("data")
OUT = Path("model")
SPLITY = [5, 10, 15, 20]
KWANTYLE = {"dol": 0.05, "p50": 0.5, "gora": 0.95}
CECHY = ["wiek", "kobieta", "czas_wej"]


def na_sekundy(s):
    if pd.isna(s):
        return np.nan
    parts = str(s).split(":")
    if len(parts) != 3:
        return np.nan
    try:
        h, m, sec = map(int, parts)
    except ValueError:
        return np.nan
    return h * 3600 + m * 60 + sec


def wczytaj():
    ramki = []
    for rok in (2023, 2024):
        d = pd.read_csv(DATA / f"halfmarathon_wroclaw_{rok}__final.csv", sep=";")
        out = pd.DataFrame({
            "rok": rok,
            "wiek": rok - d["Rocznik"],
            "kobieta": (d["Płeć"] == "K").astype(int),
            "plec": d["Płeć"],
            "meta": d["Czas"].map(na_sekundy),
        })
        for km in SPLITY:
            out[f"t{km}"] = d[f"{km} km Czas"].map(na_sekundy)
        ramki.append(out)
    df = pd.concat(ramki, ignore_index=True)
    df = df.dropna(subset=["wiek", "plec", "meta"] + [f"t{k}" for k in SPLITY])
    df = df[df["plec"].isin(["K", "M"])]
    df = df[df["wiek"].between(16, 85)]
    # Odrzucamy oczywiste błędy pomiaru: międzyczasy muszą rosnąć
    rosnie = (df["t5"] < df["t10"]) & (df["t10"] < df["t15"]) & (df["t15"] < df["t20"]) & (df["t20"] < df["meta"])
    df = df[rosnie & df["meta"].between(60 * 60, 4 * 3600)]
    return df.reset_index(drop=True)


def model(loss="absolute_error", q=None):
    kw = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
              min_samples_leaf=40, monotonic_cst=[0, 0, 1], random_state=42)
    if loss == "quantile":
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw)
    return HistGradientBoostingRegressor(loss=loss, **kw)


def zbuduj(df, wej_km):
    X = df[CECHY[:2]].assign(czas_wej=df[f"t{wej_km}"])
    modele = {"meta": {}, "splity": {}}
    for nazwa, q in KWANTYLE.items():
        modele["meta"][nazwa] = model("quantile", q).fit(X, df["meta"])
    for km in SPLITY:
        if km > wej_km:
            modele["splity"][km] = model().fit(X, df[f"t{km}"])
    return modele


def ocen(df_tr, df_te, wej_km):
    m = zbuduj(df_tr, wej_km)
    X = df_te[CECHY[:2]].assign(czas_wej=df_te[f"t{wej_km}"])
    y = df_te["meta"].to_numpy()
    p10, p50, p90 = (m["meta"][k].predict(X) for k in ("dol", "p50", "gora"))
    # Punkt odniesienia: prosta formuła Riegela T2 = T1 * (D2/D1)^1.06
    riegel = df_te[f"t{wej_km}"].to_numpy() * (21.0975 / wej_km) ** 1.06
    return {
        "mae_min": round(mean_absolute_error(y, p50) / 60, 2),
        "mae_riegel_min": round(mean_absolute_error(y, riegel) / 60, 2),
        "pokrycie_przedzialu": round(float(np.mean((y >= p10) & (y <= p90))), 3),
        "n_test": int(len(y)),
    }


def main():
    df = wczytaj()
    tr, te = df[df["rok"] == 2023], df[df["rok"] == 2024]
    metryki = {f"{km}km": ocen(tr, te, km) for km in (5, 10)}
    metryki["n_trening_finalny"] = int(len(df))
    print(json.dumps(metryki, indent=2, ensure_ascii=False))

    pakiet = {
        "wersja": "2.0",
        "modele": {km: zbuduj(df, km) for km in (5, 10)},
        # Rozkład wyników do liczenia percentyla w grupie (płeć + dekada wieku)
        "rozklad": {
            "kobieta": df["kobieta"].to_numpy(dtype="int8"),
            "dekada": (df["wiek"] // 10 * 10).clip(20, 70).to_numpy(dtype="int16"),
            "meta": df["meta"].to_numpy(dtype="int32"),
        },
        "metryki": metryki,
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump(pakiet, OUT / "tempomat.joblib", compress=3)
    (OUT / "metryki.json").write_text(json.dumps(metryki, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Zapisano", OUT / "tempomat.joblib")


if __name__ == "__main__":
    main()
