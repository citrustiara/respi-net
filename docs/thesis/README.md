# Praca inżynierska — szkic

Cały tekst pracy jest w jednym pliku: [`praca_inzynierska.tex`](praca_inzynierska.tex).
Rysunki leżą w [`figures/`](figures/).

## Kompilacja

```bash
tectonic praca_inzynierska.tex        # zainstalowane przez: brew install tectonic
```

Plik kompiluje się też pod pdfLaTeX (np. na Overleafie) — preambuła wykrywa
silnik przez `iftex` i pod pdfLaTeX używa `inputenc`/`fontenc`/`lmodern`,
a pod XeLaTeX/tectonic `fontspec` z Latin Modern. **Nie usuwać tego bloku** —
bez niego polskie znaki diakrytyczne znikają z PDF-a przy XeTeX-u.

## Struktura

- Część I — przegląd literatury (SOTA): celowo pusty szkielet z zaznaczonymi miejscami.
- Część II — zbudowany system: HB100 + front-end analogowy, A121, tor IMU,
  **algorytmy estymacji oddechu i tętna**, eksperyment soczewki/folia
  (korzysta z tych algorytmów, więc idzie po nich), sieci neuronowe (do zrobienia).
- Część III — część eksperymentalna: sen, algorytm klasyfikacji faz snu,
  porównanie z Garmin Fenix 7.

Makro `\todo{...}` (czerwony tekst) i `\miejsce{...}` (szara ramka) oznaczają
fragmenty do uzupełnienia — usunąć definicje przed oddaniem pracy.

## Skąd pochodzą liczby

- Eksperyment soczewki/folia: `data/raw/a121/guided/analysis_metrics.csv` (36 nagrań).
- Retest A121 przy 2 m (folia × geometria): skrypt
  `tools/analyze_a121_foil_2m.py` odczytuje najnowszą sesję
  `data/raw/a121/guided/guided_foil2m_*`, zapisuje lokalnie
  `analysis_2m_metrics.csv` i `analysis_2m_summary.json` oraz odtwarza cztery
  rysunki `docs/thesis/figures/foil2m_*.png`. Surowe CSV i katalog sesji są
  celowo ignorowane przez Git.
- Wyniki nocne i porównanie z Garminem: `docs/reports/a121_sleep_lens_report.tex`
  oraz pliki `*_radar_sleep_score.json` w `data/plots/`.
- Rozdział o algorytmach: `src/respi_net/a121_vitals.py` (wzory, progi, pasma).
  Sekcja „Historia: co nie działało” odtworzona z historii repozytorium —
  commity `7924d95`, `9d2130a` („vital signals fix”) i `bf82a08`.
- Algorytm snu: `src/respi_net/a121_sleep.py` (reguły faz, ocena 0–100) oraz
  `tools/analyze_a121_sleep_vitals_gated.py` (okna 60 s / krok 30 s, bramkowanie).
- Rozdzielone wykresy faz, tętna i oddechu dla stanowiska przy krawędzi łóżka
  oraz trzech pełnych nocy z płaską pokrywą nad łóżkiem odtwarza
  `tools/plot_a121_sleep_thesis.py` z plików `*_gated_sleep_vitals.csv` w
  lokalnym archiwum. Skrypt celowo nie łączy długich luk odrzuconych estymat.
- Porównanie oceny snu i faz z Garminem (noc 1): archiwum pomiarowe
  `data/archive/measurements_2026-07-14/` (nie w gicie, 12,4 GiB) —
  `e_drive/a121/garmin_reference_*/Sleep.csv` (podsumowanie z Garmin Connect)
  oraz pliki `*_SLEEP_DATA.fit` (surowy strumień stadiów).
  **Uwaga:** te dwa źródła podają dla tej samej nocy różne fazy snu
  (REM 47 min wg FIT vs 116 min wg aplikacji) — jest to opisane w pracy
  i stanowi argument, że Garmin nie jest wzorcem.
