"""Tests for correspondence trace channels and Suite2p GUI exports."""

import sys
import types

import numpy as np
import pytest

from astroglial_morphology import correspondence
from astroglial_morphology.correspondence import _normalize_trace_channels


def test_trace_channel_count_above_two_is_rejected():
    with pytest.raises(ValueError, match="only one or two channels"):
        _normalize_trace_channels([0, 1], nchannels=3)


def test_trace_channel_index_above_one_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        _normalize_trace_channels([2], nchannels=2)


def test_two_channel_gui_export_requires_both_mean_images():
    with pytest.raises(ValueError, match="meanImg_chan2"):
        correspondence._validate_suite2p_gui_projection_images(
            {"meanImg": np.zeros((2, 3), dtype=np.float32)},
            nchannels=2,
            ly=2,
            lx=3,
        )


def _install_fake_suite2p(monkeypatch, traces):
    """Provide the small Suite2p surface used by trace-export unit tests."""

    class FakeBinaryFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    suite2p = types.ModuleType("suite2p")
    suite2p.extraction_wrapper = lambda *args, **kwargs: traces
    suite2p.classify = lambda **kwargs: np.array([[1.0, 1.0]])

    classification = types.ModuleType("suite2p.classification")
    classification.builtin_classfile = "builtin-classifier"

    extraction = types.ModuleType("suite2p.extraction")
    extraction.preprocess = lambda F, **kwargs: F
    extraction.oasis = lambda F, **kwargs: F + 100.0

    io = types.ModuleType("suite2p.io")
    io.BinaryFile = FakeBinaryFile

    monkeypatch.setitem(sys.modules, "suite2p", suite2p)
    monkeypatch.setitem(sys.modules, "suite2p.classification", classification)
    monkeypatch.setitem(sys.modules, "suite2p.extraction", extraction)
    monkeypatch.setitem(sys.modules, "suite2p.io", io)


def test_channel_one_trace_export_has_complete_suite2p_gui_layout(tmp_path, monkeypatch):
    """Channel-1-only exports must still have Suite2p's canonical trace files."""
    f_ch0 = np.full((1, 4), 10.0, dtype=np.float32)
    fneu_ch0 = np.full((1, 4), 1.0, dtype=np.float32)
    f_ch1 = np.full((1, 4), 20.0, dtype=np.float32)
    fneu_ch1 = np.full((1, 4), 2.0, dtype=np.float32)
    _install_fake_suite2p(
        monkeypatch,
        (np.array([{"ypix": [0]}], dtype=object), f_ch0, fneu_ch0, f_ch1, fneu_ch1),
    )
    monkeypatch.setattr(
        correspondence,
        "_build_suite2p_stat",
        lambda *args, **kwargs: np.array([{"ypix": [0]}], dtype=object),
    )

    ops = {
        "nchannels": 2,
        "Ly": 2,
        "Lx": 3,
        "nframes": 4,
        "neucoeff": 0.7,
        "baseline": "maximin",
        "win_baseline": 60.0,
        "sig_baseline": 10.0,
        "fs": 2.0,
        "prctile_baseline": 8.0,
        "batch_size": 10,
        "tau": 1.0,
        "meanImg": np.ones((2, 3), dtype=np.float32),
        "meanImg_chan2": np.full((2, 3), 2.0, dtype=np.float32),
        "reg_file": str(tmp_path / "data.bin"),
        "reg_file_chan2": str(tmp_path / "data_chan2.bin"),
    }
    np.save(tmp_path / "ops.npy", ops)
    np.save(tmp_path / "subsegmented_masks_seg.npy", {"masks": np.ones((2, 3))})
    (tmp_path / "data.bin").touch()
    (tmp_path / "data_chan2.bin").touch()

    output_dir, extracted = correspondence._extract_suite2p_traces_for_channels(
        data_path=tmp_path,
        mask_filename="subsegmented_masks_seg.npy",
        trace_channels=[1],
    )

    assert set(extracted) == {1}
    np.testing.assert_array_equal(extracted[1], f_ch1)
    for filename in (
        "stat.npy",
        "ops.npy",
        "iscell.npy",
        "F.npy",
        "Fneu.npy",
        "spks.npy",
        "F_chan2.npy",
        "Fneu_chan2.npy",
        "redcell.npy",
        "F_ch1.npy",
        "Fneu_ch1.npy",
        "spks_ch1.npy",
    ):
        assert (output_dir / filename).is_file(), filename

    # The requested source channel becomes the GUI's primary trace, while the
    # other acquisition channel remains available through Suite2p's standard
    # second-channel files.
    np.testing.assert_array_equal(np.load(output_dir / "F.npy"), f_ch1)
    np.testing.assert_array_equal(np.load(output_dir / "Fneu.npy"), fneu_ch1)
    np.testing.assert_array_equal(np.load(output_dir / "F_chan2.npy"), f_ch0)
    np.testing.assert_array_equal(np.load(output_dir / "Fneu_chan2.npy"), fneu_ch0)
    np.testing.assert_array_equal(
        np.load(output_dir / "spks.npy"), f_ch1 - 0.7 * fneu_ch1 + 100.0
    )

    gui_ops = np.load(output_dir / "ops.npy", allow_pickle=True).item()
    assert gui_ops["functional_chan"] == 1
    assert gui_ops["astroglial_source_trace_channel"] == 1
    assert gui_ops["channel_indices"] == [1, 0]
    assert gui_ops["align_by_chan"] == 2
    assert gui_ops["reg_file"] == str(tmp_path / "data_chan2.bin")
    assert gui_ops["reg_file_chan2"] == str(tmp_path / "data.bin")
    np.testing.assert_array_equal(gui_ops["meanImg"], ops["meanImg_chan2"])
    np.testing.assert_array_equal(gui_ops["meanImg_chan2"], ops["meanImg"])
    np.testing.assert_array_equal(
        np.load(output_dir / "redcell.npy"), np.zeros((1, 2))
    )


def test_single_channel_gui_export_has_no_second_channel_artifacts(tmp_path):
    stat = np.array([{"ypix": [0]}], dtype=object)
    traces = np.arange(4, dtype=np.float32).reshape(1, 4)
    neuropil = np.ones((1, 4), dtype=np.float32)
    spikes = np.full((1, 4), 3.0, dtype=np.float32)
    data_bin = tmp_path / "data.bin"
    data_bin.touch()

    correspondence._save_suite2p_gui_output(
        output_dir=tmp_path / "cellpose_suite2p_output",
        ops={"Ly": 2, "Lx": 2, "nframes": 4, "fs": 2.0, "tau": 1.0},
        stat=stat,
        iscell=np.array([[1.0, 1.0]]),
        channel_traces={0: (traces, neuropil)},
        channel_spikes={0: spikes},
        selected_channels=[0],
        nchannels=1,
        data_bin=data_bin,
        data_chan2_bin=None,
    )

    output_dir = tmp_path / "cellpose_suite2p_output"
    for filename in (
        "stat.npy",
        "ops.npy",
        "iscell.npy",
        "F.npy",
        "Fneu.npy",
        "spks.npy",
    ):
        assert (output_dir / filename).is_file(), filename
    assert not (output_dir / "F_chan2.npy").exists()
    assert not (output_dir / "Fneu_chan2.npy").exists()
    assert not (output_dir / "redcell.npy").exists()
