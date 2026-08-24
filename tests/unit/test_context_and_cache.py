from hashlib import sha256

import numpy as np

from r680_safety_planner.data import FeatureCacheRecord, load_feature_cache, save_feature_cache
from r680_safety_planner.planning.context import occupancy_to_channels, resample_route, transform_xy


def test_route_and_costmap_context() -> None:
    route = resample_route(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), 0.5, 1.5)
    assert route.shape == (4, 2)
    local = transform_xy(route, np.array([0.5, 0.0]), 0.0)
    assert np.isclose(local[0, 0], -0.5)
    channels = occupancy_to_channels(np.array([[-1, 0, 100]]))
    assert channels.shape == (3, 1, 3)
    assert np.all(channels.sum(axis=0) == 1)


def test_feature_cache_round_trip(tmp_path) -> None:
    digest = sha256(b"test").hexdigest()
    record = FeatureCacheRecord(np.ones((2, 3, 4)), 1.25, "laser", digest, digest)
    path = tmp_path / "feature.npz"
    save_feature_cache(path, record)
    loaded = load_feature_cache(path)
    assert loaded.frame_id == "laser"
    assert loaded.bev_feature.shape == (2, 3, 4)
