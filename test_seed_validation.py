"""seed CSV 検証（SeedDataError）のテスト。リポジトリ内の data/ は変更しない。"""

import csv
from pathlib import Path

import pytest

from ollama_hero_gen import Config, LocalStorage, SeedDataError


def _make_config(tmp_path: Path) -> Config:
    return Config(
        model="gpt-oss:20b",
        host="http://localhost:11434",
        data_dir=str(tmp_path),
        num_iterations=1,
    )


def _write_seed(tmp_path: Path, key: str, rows):
    path = tmp_path / f"seed_{key}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([key])
        for row in rows:
            writer.writerow([row] if isinstance(row, str) else row)
    return path


def test_valid_seed_files_initialize(tmp_path):
    """正常な seed CSV から従来どおりランダムに値を取得できる"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA", "ValueB"])
    storage = LocalStorage(_make_config(tmp_path))
    assert storage.get_random_attribute("age") in {"ValueA", "ValueB"}


def test_missing_seed_file_created_from_defaults(tmp_path):
    """不在の seed CSV は初期データで作成される"""
    storage = LocalStorage(_make_config(tmp_path))
    for key, path in storage.seed_files.items():
        assert path.exists()
        assert storage.get_random_attribute(key)  # 空ではない


def test_invalid_header_raises_seed_data_error(tmp_path):
    """ヘッダー不正では SeedDataError が発生し、ファイル名と期待ヘッダーが表示される"""
    _write_seed(tmp_path, "age", ["ValueA"])
    path = tmp_path / "seed_age.csv"
    # ヘッダーを書き換える
    path.write_text("wrong_header\nValueA\n", encoding="utf-8")
    with pytest.raises(SeedDataError) as exc:
        LocalStorage(_make_config(tmp_path))
    assert "seed_age.csv" in str(exc.value)
    assert "age" in str(exc.value)


def test_empty_file_raises_seed_data_error(tmp_path):
    """空ファイルでは SeedDataError が発生する"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA"])
    (tmp_path / "seed_age.csv").write_text("", encoding="utf-8")
    with pytest.raises(SeedDataError):
        LocalStorage(_make_config(tmp_path))


def test_blank_lines_only_raises_seed_data_error(tmp_path):
    """空行だけのファイルでは SeedDataError が発生する"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA"])
    (tmp_path / "seed_age.csv").write_text("age\n\n\n", encoding="utf-8")
    with pytest.raises(SeedDataError):
        LocalStorage(_make_config(tmp_path))


def test_multi_column_row_raises_seed_data_error(tmp_path):
    """2列以上のデータ行では SeedDataError が発生する"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA"])
    path = tmp_path / "seed_age.csv"
    path.write_text("age\none,two\n", encoding="utf-8")
    with pytest.raises(SeedDataError) as exc:
        LocalStorage(_make_config(tmp_path))
    assert "multi-column" in str(exc.value)


def test_no_valid_values_raises_seed_data_error(tmp_path):
    """有効な値が0件の場合、IndexErrorではなくSeedDataErrorが発生する"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA"])
    (tmp_path / "seed_age.csv").write_text("age\n   \n", encoding="utf-8")
    with pytest.raises(SeedDataError):
        LocalStorage(_make_config(tmp_path))


def test_blank_rows_are_skipped(tmp_path):
    """空行が混在していても、空文字列を属性として選ばない"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA"])
    (tmp_path / "seed_age.csv").write_text(
        "age\n\n  \nRealValue\n", encoding="utf-8"
    )
    storage = LocalStorage(_make_config(tmp_path))
    assert storage.get_random_attribute("age") == "RealValue"


def test_append_seed_rejects_empty_value(tmp_path):
    """append_seed() に空文字列を渡すと SeedDataError が発生する"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA"])
    storage = LocalStorage(_make_config(tmp_path))
    with pytest.raises(SeedDataError):
        storage.append_seed("age", "")
    with pytest.raises(SeedDataError):
        storage.append_seed("age", "   ")


def test_append_seed_trims_whitespace(tmp_path):
    """append_seed() は前後の空白を除去して保存する"""
    for key in ["age", "gender", "species", "ability", "wants", "role"]:
        _write_seed(tmp_path, key, ["ValueA"])
    storage = LocalStorage(_make_config(tmp_path))
    storage.append_seed("age", "  NewValue  ")
    assert "NewValue" in storage._read_seed_values(storage.seed_files["age"])
