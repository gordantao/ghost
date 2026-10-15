import numpy as np

from ghost.data import game_folds, split_indices


def test_folds_keep_games_together():
    game_id = np.repeat(np.arange(20), 7)
    folds = game_folds(game_id, n_folds=5, seed=0)
    for g in np.unique(game_id):
        assert len(set(folds[game_id == g])) == 1
    assert set(folds) == set(range(5))


def test_split_is_disjoint_by_game():
    game_id = np.repeat(np.arange(30), 5)
    train, val, held = split_indices(game_id, fold=2, n_folds=5, val_frac=0.1, seed=0)
    assert len(train) + len(val) + len(held) == len(game_id)
    games = [set(game_id[i]) for i in (train, val, held)]
    assert not (games[0] & games[1] or games[0] & games[2] or games[1] & games[2])

    train, val, held = split_indices(game_id, fold=None, n_folds=5, val_frac=0.1, seed=0)
    assert len(held) == 0 and len(train) + len(val) == len(game_id)
