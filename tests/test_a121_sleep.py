from respi_net.a121_sleep import _bridge_direct_deep_rem_transitions


def test_direct_rem_to_deep_transition_gets_light_bridge() -> None:
    phases = ["awake", "rem", "rem", "deep", "deep", "deep", "deep", "deep", "light"]

    bridged, corrected = _bridge_direct_deep_rem_transitions(
        phases,
        min_len=3,
        sleep_start=1,
        sleep_end=8,
    )

    assert bridged == ["awake", "rem", "rem", "light", "light", "light", "deep", "deep", "light"]
    assert corrected == [False, False, False, True, True, True, False, False, False]


def test_direct_deep_to_rem_transition_gets_light_bridge() -> None:
    phases = ["deep", "deep", "rem", "rem", "light"]

    bridged, corrected = _bridge_direct_deep_rem_transitions(
        phases,
        min_len=2,
        sleep_start=0,
        sleep_end=4,
    )

    assert bridged == ["deep", "deep", "light", "light", "light"]
    assert sum(corrected) == 2


def test_existing_light_transition_is_left_unchanged() -> None:
    phases = ["deep", "deep", "light", "rem", "rem"]

    bridged, corrected = _bridge_direct_deep_rem_transitions(
        phases,
        min_len=3,
        sleep_start=0,
        sleep_end=4,
    )

    assert bridged == phases
    assert not any(corrected)


def test_a121_capture_retries_the_connection(monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("acconeer.exptool")
    from respi_net import a121

    answers = iter([False, True])
    calls = []

    def once(self, client_module, get_distances, port):
        calls.append(port)
        return next(answers)

    monkeypatch.setattr(a121.A121Capture, "_connect_once", once)
    monkeypatch.setattr(a121, "A121_CONNECT_RETRY_PAUSE_S", 0.0)
    capture = a121.A121Capture(config=a121.A121Config())
    assert capture.connect("/dev/cu.test") and calls == ["/dev/cu.test", "/dev/cu.test"]
