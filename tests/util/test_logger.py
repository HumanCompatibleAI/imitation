"""Tests `imitation.util.logger`."""

import csv
import json
import os.path as osp
from collections import defaultdict

import pytest
from stable_baselines3.common import logger as sb_logger

import imitation.util.logger as logger


def _csv_to_dict(csv_path: str) -> dict:
    result = defaultdict(list)
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if v != "":
                    v = float(v)
                result[k].append(v)
    return result


def _json_to_dict(json_path: str) -> dict:
    r"""Loads the saved json logging file and convert it to expected dict format.

    Args:
        json_path: Path of the json log file.
            Stored in the format - '{"A": 1, "B": 1}\n{"A": 2}\n{"B": 3}\n'

    Returns:
        dictionary in the format - `{"A": [1, 2, ""], "B": [1, "", 3]}`
    """
    result = defaultdict(list)
    with open(json_path, "r") as f:
        all_line_dicts = [json.loads(line) for line in f.readlines()]
    # get all the keys in the dict so as to add "" if the key is not present in a line
    all_keys = set().union(*[list(line_dict.keys()) for line_dict in all_line_dicts])

    for line_dict in all_line_dicts:
        for key in all_keys:
            result[key].append(line_dict.get(key, ""))
    return result


def _compare_csv_lines(csv_path: str, expect: dict):
    observed = _csv_to_dict(csv_path)
    assert expect == observed


def _compare_json_lines(json_path: str, expect: dict):
    observed = _json_to_dict(json_path)
    assert expect == observed


def test_no_accum(tmpdir):
    hier_logger = logger.configure(tmpdir, ["csv", "json"])
    assert hier_logger.get_dir() == tmpdir

    # Check that the recorded "A": -1 is overwritten by "A": 1 in the next line.
    # Previously, the observed value would be the mean of these two values (0) instead.
    hier_logger.record("A", -1)
    hier_logger.record("A", 1)
    hier_logger.record("B", 1)
    hier_logger.dump()

    hier_logger.record("A", 2)
    hier_logger.dump()
    hier_logger.record("B", 3)
    hier_logger.dump()
    expect = {"A": [1, 2, ""], "B": [1, "", 3]}
    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect)
    _compare_json_lines(osp.join(tmpdir, "progress.json"), expect)


def test_raise_unknown_format():
    with pytest.raises(ValueError, match=r"Unknown format specified:.*"):
        logger.make_output_format("txt", "log_dir")


def test_free_form(tmpdir):
    hier_logger = logger.configure(tmpdir, ["log"])

    hier_logger.log("info 1")
    hier_logger.info("info 2")
    hier_logger.warn("warn 1")
    hier_logger.error("error 1")
    hier_logger.debug("debug not printed")
    hier_logger.set_level(level=sb_logger.DEBUG)
    hier_logger.debug("debug 1")
    with hier_logger.accumulate_means("foo"):
        hier_logger.info("info inner")
    hier_logger.debug("debug outer")

    hier_logger.close()

    with open(osp.join(tmpdir, "log.txt"), "r") as f:
        contents = f.readlines()
    assert contents == [
        "info 1\n",
        "info 2\n",
        "warn 1\n",
        "error 1\n",
        "debug 1\n",
        "info inner\n",
        "debug outer\n",
    ]


def test_reentry_fails(tmpdir):
    hier_logger = logger.configure(tmpdir)

    with hier_logger.accumulate_means("foo"):
        with pytest.raises(RuntimeError, match=r"Nested.*"):
            with hier_logger.accumulate_means("bar"):  # pragma: no cover
                pass


def test_close(tmpdir):
    hier_logger = logger.configure(tmpdir)

    hier_logger.record("A", 1)
    with hier_logger.accumulate_means("foo"):
        hier_logger.record("B", 2)
    hier_logger.dump()

    hier_logger.close()
    hier_logger.record("foo", 42)
    with pytest.raises(ValueError, match="I/O operation on closed file."):
        hier_logger.dump()


def test_name_to_value(tmpdir):
    hier_logger = logger.configure(tmpdir)

    with hier_logger.accumulate_means("foo"):
        hier_logger.record("A", 1)
        assert hier_logger.name_to_value["raw/foo/A"] == 1
        assert hier_logger.name_to_count["raw/foo/A"] == 0
        assert hier_logger.name_to_excluded["raw/foo/A"] is hier_logger.to_tuple(None)
        hier_logger.record("B", 10)
        assert hier_logger.name_to_value["raw/foo/B"] == 10
        assert hier_logger.name_to_count["raw/foo/B"] == 0
        assert hier_logger.name_to_excluded["raw/foo/B"] is hier_logger.to_tuple(None)
        hier_logger.dump()
        hier_logger.record("B", 20)
        assert hier_logger.name_to_value["raw/foo/B"] == 20
        assert hier_logger.name_to_count["raw/foo/B"] == 0
        assert hier_logger.name_to_excluded["raw/foo/B"] is hier_logger.to_tuple(None)
    assert hier_logger.name_to_value["mean/foo/A"] == 1
    assert hier_logger.name_to_count["mean/foo/A"] == 1
    assert hier_logger.name_to_excluded["mean/foo/A"] is hier_logger.to_tuple(None)
    assert hier_logger.name_to_value["mean/foo/B"] == 15
    assert hier_logger.name_to_count["mean/foo/B"] == 2
    assert hier_logger.name_to_excluded["mean/foo/B"] is hier_logger.to_tuple(None)
    hier_logger.dump()
    assert len(hier_logger.name_to_value) == 0
    assert len(hier_logger.name_to_count) == 0
    assert len(hier_logger.name_to_excluded) == 0


def test_hard(tmpdir):
    hier_logger = logger.configure(tmpdir)

    # Part One: Test logging outside the accumulating scope, and within scopes
    # with two different logging keys (including a repeat).

    hier_logger.record("no_context", 1)

    with hier_logger.accumulate_means("disc"):
        hier_logger.record("C", 2)
        hier_logger.record("D", 2)
        hier_logger.dump()
        hier_logger.record("C", 4)
        hier_logger.dump()

    with hier_logger.accumulate_means("gen"):
        hier_logger.record("E", 2)
        hier_logger.dump()
        hier_logger.record("E", 0)
        hier_logger.dump()

    with hier_logger.accumulate_means("disc"):
        hier_logger.record("C", 3)
        hier_logger.dump()

    hier_logger.dump()  # Writes 1 mean each from "gen" and "disc".

    expect_raw_gen = {"raw/gen/E": [2, 0]}
    expect_raw_disc = {
        "raw/disc/C": [2, 4, 3],
        "raw/disc/D": [2, "", ""],
    }
    expect_default = {
        "mean/gen/E": [1],
        "mean/disc/C": [3],
        "mean/disc/D": [2],
        "no_context": [1],
    }

    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect_default)
    _compare_csv_lines(osp.join(tmpdir, "raw", "gen", "progress.csv"), expect_raw_gen)
    _compare_csv_lines(osp.join(tmpdir, "raw", "disc", "progress.csv"), expect_raw_disc)

    # Part Two:
    # Check that we append to the same logs after the first dump to "means/*".

    with hier_logger.accumulate_means("disc"):
        hier_logger.record("D", 100)
        hier_logger.dump()

    hier_logger.record("no_context", 2)

    hier_logger.dump()  # Writes 1 mean from "disc". "gen" is blank.

    expect_raw_gen = {"raw/gen/E": [2, 0]}
    expect_raw_disc = {
        "raw/disc/C": [2, 4, 3, ""],
        "raw/disc/D": [2, "", "", 100],
    }
    expect_default = {
        "mean/gen/E": [1, ""],
        "mean/disc/C": [3, ""],
        "mean/disc/D": [2, 100],
        "no_context": [1, 2],
    }

    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect_default)
    _compare_csv_lines(osp.join(tmpdir, "raw", "gen", "progress.csv"), expect_raw_gen)
    _compare_csv_lines(osp.join(tmpdir, "raw", "disc", "progress.csv"), expect_raw_disc)


def test_accumulate_prefix(tmpdir):
    hier_logger = logger.configure(tmpdir)

    with hier_logger.add_accumulate_prefix("foo"), hier_logger.accumulate_means("bar"):
        hier_logger.record("A", 1)
        hier_logger.record("B", 2)
        hier_logger.dump()

    hier_logger.record("no_context", 1)

    with hier_logger.accumulate_means("blat"):
        hier_logger.record("C", 3)
        hier_logger.dump()

    hier_logger.dump()

    expect_raw_foo_bar = {
        "raw/foo/bar/A": [1],
        "raw/foo/bar/B": [2],
    }
    expect_raw_blat = {
        "raw/blat/C": [3],
    }
    expect_default = {
        "mean/foo/bar/A": [1],
        "mean/foo/bar/B": [2],
        "mean/blat/C": [3],
        "no_context": [1],
    }

    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect_default)
    _compare_csv_lines(
        osp.join(tmpdir, "raw", "foo", "bar", "progress.csv"),
        expect_raw_foo_bar,
    )
    _compare_csv_lines(osp.join(tmpdir, "raw", "blat", "progress.csv"), expect_raw_blat)


def test_key_prefix(tmpdir):
    hier_logger = logger.configure(tmpdir)

    with hier_logger.accumulate_means("foo"), hier_logger.add_key_prefix("bar"):
        hier_logger.record("A", 1)
        hier_logger.record("B", 2)
        hier_logger.dump()

    hier_logger.record("no_context", 1)

    with hier_logger.accumulate_means("blat"):
        hier_logger.record("C", 3)
        hier_logger.dump()

    hier_logger.dump()

    expect_raw_foo_bar = {
        "raw/foo/bar/A": [1],
        "raw/foo/bar/B": [2],
    }
    expect_raw_blat = {
        "raw/blat/C": [3],
    }
    expect_default = {
        "mean/foo/bar/A": [1],
        "mean/foo/bar/B": [2],
        "mean/blat/C": [3],
        "no_context": [1],
    }

    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect_default)
    _compare_csv_lines(
        osp.join(tmpdir, "raw", "foo", "progress.csv"),
        expect_raw_foo_bar,
    )
    _compare_csv_lines(osp.join(tmpdir, "raw", "blat", "progress.csv"), expect_raw_blat)


def test_cant_add_prefix_within_accumulate_means(tmpdir):
    h = logger.configure(tmpdir)
    with pytest.raises(RuntimeError):
        with h.accumulate_means("foo"), h.add_accumulate_prefix("bar"):
            pass  # pragma: no cover


def test_cant_add_key_prefix_outside_accumulate_means(tmpdir):
    h = logger.configure(tmpdir)
    with pytest.raises(RuntimeError):
        with h.add_key_prefix("bar"):
            pass  # pragma: no cover
