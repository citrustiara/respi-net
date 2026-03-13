import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from script import BreathCapture

@pytest.fixture
def csv_path():
    # return "respiratory_6axis_raw_2026-03-08_02-54-20.csv"
    # return "respiratory_6axis_raw_2026-03-08_02-54-20.csv"
    # return "respiratory_6axis_raw_2026-03-08_15-08-44.csv"
    return "respiratory_6axis_raw_2026-03-08_02-37-19.csv"
    # return "respiratory_6axis_raw_2026-03-08_03-01-12.csv"


def test_stop_and_graph(csv_path):
    # Wczytujemy dane testowe
    df = pd.read_csv(csv_path)
    
    app = BreathCapture()
    # Przekształcamy DataFrame na listę list (format oczekiwany przez script.py)
    app.data_storage = df[['Time_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.tolist()
    
    # Patchujemy to_csv i plt.show, aby test nie śmiecił plikami i nie otwierał okien
    with patch("pandas.DataFrame.to_csv"):
    # with patch("pandas.DataFrame.to_csv"), \
    #      patch("matplotlib.pyplot.show"), \
    #      patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())):
        app.stop_and_graph()
    
    # Weryfikujemy czy analiza coś wypluła
    print(f"\nTest Result: FS={app.fs:.1f} Hz, Resp={app.resp_bpm:.1f} BPM, Heart={app.heart_bpm:.1f} BPM")
    
    assert app.fs > 100
    # Sprawdzamy czy wartości są w sensownych zakresach (oddech może być bardzo wolny)
    assert 0 <= app.resp_bpm <= 50
    # Sprawdzamy czy tętno wyliczone z FFT również wpada w prawidłowy przedział
    assert 40 <= app.heart_bpm <= 240
