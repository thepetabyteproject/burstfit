import pytest

from burstfit.utils.functions import *


def test_gauss():
    x = np.linspace(0, 99, 100)
    g = gauss(x, 10, 30, 20)
    assert pytest.approx(np.trapz(g), rel=0.1) == 10
    assert x[np.argmax(g)] == 30


def test_gauss_norm():
    x = np.linspace(0, 99, 100)
    g = gauss_norm(x, 30, 20)
    assert pytest.approx(np.trapz(g), rel=0.1) == 1
    assert x[np.argmax(g)] == 30


def test_gauss_norm2():
    x = np.linspace(0, 99, 100)
    g = gauss_norm2(x, 30, 20, 70, 10, 0.9)
    assert pytest.approx(np.trapz(g), rel=0.1) == 1
    assert x[np.argmax(g)] == 30

    g = gauss_norm2(x, 30, 20, 70, 10, 0.1)
    assert pytest.approx(np.trapz(g), rel=0.1) == 1
    assert x[np.argmax(g)] == 70


def test_gauss_norm3():
    x = np.linspace(0, 99, 100)
    g = gauss_norm3(x, 20, 10, 70, 10, 80, 5, 0.9, 0.05)
    assert pytest.approx(np.trapz(g), rel=0.1) == 1
    assert x[np.argmax(g)] == 20

    g = gauss_norm3(x, 20, 10, 70, 10, 80, 5, 0.05, 0.9)
    assert pytest.approx(np.trapz(g), rel=0.1) == 1
    assert pytest.approx(x[np.argmax(g)], rel=1) == 70

    g = gauss_norm3(x, 20, 10, 70, 10, 80, 5, 0.05, 0.1)
    assert pytest.approx(np.trapz(g), rel=0.1) == 1
    assert pytest.approx(x[np.argmax(g)], rel=1) == 80


def test_pulse_fn():
    x = np.linspace(0, 99, 100)
    p = pulse_fn(x, -1, 10, 10, 10)
    assert p.all() == 0

    p = pulse_fn(x, 10, 50, 20, 2)
    assert pytest.approx(np.trapz(p), rel=0.1) == 10
    assert pytest.approx(x[np.argmax(p)], rel=1) == 20

    p = pulse_fn(x, 10, 50, 10, 5)
    assert pytest.approx(np.trapz(p), rel=0.1) == 10
    assert pytest.approx(x[np.argmax(p)], rel=1) == 20


def test_pulse_fn_vec():
    x = np.linspace(0, 99, 100)
    p = pulse_fn_vec(x, 10, 50, 20, 2)
    assert pytest.approx(np.trapz(p), rel=0.1) == 10
    assert pytest.approx(x[np.argmax(p)], rel=1) == 20

    p = pulse_fn_vec(x, 10, 50, 10, 5)
    assert pytest.approx(np.trapz(p), rel=0.1) == 10
    assert pytest.approx(x[np.argmax(p)], rel=1) == 20

    mus = np.array([50, 20])
    taus = np.array([5, 10])
    mus = np.expand_dims(mus, -1)
    taus = np.expand_dims(taus, -1)
    p = pulse_fn_vec(x, 10, mus, 10, taus)

    assert pytest.approx(np.trapz(p[0]), rel=0.1) == 10
    assert pytest.approx(x[np.argmax(p[0])], rel=1) == 50

    assert pytest.approx(np.trapz(p[1]), rel=0.1) == 10
    assert pytest.approx(x[np.argmax(p[1])], rel=1) == 20
