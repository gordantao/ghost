import numpy as np
import pytest

pytest.importorskip('matplotlib')

from ghost.animate import render, select_play  # noqa: E402
from ghost.preprocess import N_ENT  # noqa: E402

GAME = np.array([1, 1, 2, 2])
PLAY = np.array([10, 20, 10, 30])


def test_select_play_by_id():
    rng = np.random.default_rng(0)
    assert select_play(GAME, PLAY, None, 20, rng) == 1
    assert select_play(GAME, PLAY, 2, 10, rng) == 2


def test_select_play_needs_game_when_play_id_repeats():
    with pytest.raises(ValueError, match='several games'):
        select_play(GAME, PLAY, None, 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match='no play'):
        select_play(GAME, PLAY, 1, 30, np.random.default_rng(0))


def test_select_play_random_is_seeded_and_respects_game():
    picks = {select_play(GAME, PLAY, None, None, np.random.default_rng(s)) for s in range(20)}
    assert picks == {0, 1, 2, 3}
    assert {select_play(GAME, PLAY, 2, None, np.random.default_rng(s)) for s in range(20)} == {2, 3}
    assert (select_play(GAME, PLAY, None, None, np.random.default_rng(5))
            == select_play(GAME, PLAY, None, None, np.random.default_rng(5)))


def test_render_writes_one_gif_frame_per_play_frame(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    T, K = 4, 3
    play = {'dyn': rng.normal(0, 8, (T, N_ENT, 2)), 'ghost': rng.normal(0, 8, (K, T, 11, 2))}
    out = tmp_path / 'play.gif'
    render(play, obs_frames=1, title='t', subtitle='s', labels={0: 'WR  H +0.10'}, first_down=10.0, out=out)
    assert Image.open(out).n_frames == T
