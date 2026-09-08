"""Use Cellpose's real GUI file loader/saver without launching Qt widgets."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from cellpose.gui import io as cellpose_gui_io
from cellpose.utils import masks_to_outlines

from astroglial_morphology.gui.services.results import save_seg_masks, load_seg_file


def sparse_masks():
    masks = np.zeros((24, 40), dtype=np.int32)
    masks[2:7, 2:7] = 5
    masks[9:14, 12:17] = 9
    masks[16:21, 26:31] = 42
    return masks


def gui_parent(monkeypatch):
    """Replace widget-dependent image setup; keep GUI mask loading intact."""
    def initialize_images(parent, image, load_3D=False):
        assert image.ndim == 2
        parent.NZ = 1
        parent.Ly, parent.Lx = image.shape
        parent.Ly0, parent.Lx0 = image.shape
        parent.stack = np.repeat(image[None, ..., None], 3, axis=-1)

    monkeypatch.setattr(cellpose_gui_io, "_initialize_images", initialize_images)
    noop = lambda *args, **kwargs: None
    chooser = lambda: SimpleNamespace(setCurrentIndex=noop, currentIndex=lambda: 0)
    return SimpleNamespace(
        reset=noop, set_restore_button=noop, set_normalize_params=noop,
        draw_layer=noop, toggle_mask_ops=noop, enable_buttons=noop,
        update_layer=noop, ViewDropDown=chooser(),
        ChannelChoose=[chooser(), chooser()],
        colormap=np.tile(np.array([[90, 140, 210]], dtype=np.uint8), (256, 1)),
        track_changes=[], flows=[[], [], [], [], [[]]], color=0, diameter=30.,
        get_thresholds=lambda: (0.4, 0.),
        get_normalize_params=lambda: {"normalize": True},
    )


def test_sparse_edits_load_and_resave_with_actual_cellpose_gui_io(tmp_path, monkeypatch):
    projection = tmp_path / "projection.png"
    image = np.arange(24 * 40, dtype=np.uint8).reshape(24, 40)
    Image.fromarray(image).save(projection)
    path = tmp_path / "projection_seg.npy"
    template = {
        "masks": np.zeros((24, 40), dtype=np.int32),
        "outlines": np.ones((24, 40), dtype=np.int32),
        "filename": str(projection),
        "ismanual": np.zeros(1, bool), "zdraw": [None] * 20,
        "colors": np.ones((1, 3), dtype=np.uint8), "chan_choose": [0, 0],
    }
    np.save(path, template)
    original_bytes = path.read_bytes()
    masks = sparse_masks()
    save_seg_masks(path, masks)
    saved = load_seg_file(path)
    assert np.unique(saved["masks"]).tolist() == [0, 1, 2, 3]
    for cell, old in enumerate([5, 9, 42], 1):
        np.testing.assert_array_equal(saved["masks"] == cell, masks == old)
    np.testing.assert_array_equal(saved["outlines"], masks_to_outlines(saved["masks"]) * saved["masks"])
    assert saved["ismanual"].tolist() == [True, True, True]
    assert len(saved["zdraw"]) == 3
    assert "colors" not in saved
    assert path.with_suffix(".npy.orig").read_bytes() == original_bytes

    def unexpected_renumber(*args, **kwargs):
        pytest.fail("Cellpose should not need to renumber the edited masks")

    monkeypatch.setattr(cellpose_gui_io.fastremap, "renumber", unexpected_renumber)
    parent = gui_parent(monkeypatch)
    # Includes Cellpose's real image lookup, _masks_to_gui, and metadata loader.
    cellpose_gui_io._load_seg(parent, filename=str(path))
    assert parent.loaded and parent.ncells == 3
    np.testing.assert_array_equal(parent.cellpix[0], saved["masks"])
    np.testing.assert_array_equal(parent.outpix[0], saved["outlines"])
    np.testing.assert_array_equal(parent.ismanual, saved["ismanual"])
    assert len(parent.cellcolors) == 4
    assert len(parent.zdraw) == 3
    cellpose_gui_io._save_sets(parent)
    resaved = load_seg_file(path)
    for field in ["masks", "outlines", "ismanual"]:
        np.testing.assert_array_equal(resaved[field], saved[field])
    assert resaved["colors"].shape == (3, 3)


def test_deleted_cell_remaps_colors_and_preserves_unchanged_manual_flags(tmp_path):
    path = tmp_path / "edits_seg.npy"
    before = np.zeros((20, 20), dtype=np.int32)
    before[1:5, 1:5] = 1
    before[7:11, 7:11] = 2
    before[13:17, 13:17] = 3
    colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    np.save(path, {"masks": before, "ismanual": [True, False, False], "colors": colors})
    edited = before.copy()
    edited[edited == 1] = 0
    edited[17, 13:17] = 3
    save_seg_masks(path, edited)
    saved = load_seg_file(path)
    assert saved["ismanual"].tolist() == [False, True]
    np.testing.assert_array_equal(saved["colors"], colors[1:])
    assert saved["masks"][8, 8] == 1
    assert saved["masks"][17, 14] == 2


@pytest.mark.parametrize("masks", [np.zeros((5, 5), np.int32), np.ones((5, 5), np.int32) * 42])
def test_empty_and_full_foreground_masks(tmp_path, masks):
    path = tmp_path / "edge_seg.npy"
    save_seg_masks(path, masks)
    saved = load_seg_file(path)
    np.testing.assert_array_equal(saved["masks"], (masks > 0).astype(np.int32))
    assert len(saved["ismanual"]) == int(np.any(masks))
    assert saved["outlines"].shape == masks.shape


@pytest.mark.parametrize("masks", [np.zeros((2, 3, 4), int), np.full((3, 4), -1), np.ones((3, 4)) * .5])
def test_invalid_masks_leave_original_file_untouched(tmp_path, masks):
    path = tmp_path / "original_seg.npy"
    np.save(path, {"masks": np.zeros((3, 4), np.int32)})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        save_seg_masks(path, masks)
    assert path.read_bytes() == original
    assert not path.with_suffix(".npy.orig").exists()
