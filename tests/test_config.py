from pathlib import Path

import pytest
import yaml

from ollafleet.config import find_config, load_config, parse_nodes


def test_load_and_parse_nodes(tmp_path: Path) -> None:
    config_path = tmp_path / "fleet.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {
                        "name": "studio-mac",
                        "host": "http://127.0.0.1:11434/",
                        "tags": ["local"],
                    }
                ]
            }
        )
    )

    data = load_config(config_path)
    nodes = parse_nodes(data)

    assert find_config(str(config_path)) == config_path
    assert len(nodes) == 1
    assert nodes[0].name == "studio-mac"
    assert nodes[0].host == "http://127.0.0.1:11434"
    assert nodes[0].tags == ["local"]


def test_load_config_rejects_missing_nodes(tmp_path: Path) -> None:
    config_path = tmp_path / "fleet.yml"
    config_path.write_text("timeout: 5\n")

    with pytest.raises(ValueError, match="must have a 'nodes' list"):
        load_config(config_path)


def test_find_config_rejects_missing_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config not found"):
        find_config(str(tmp_path / "missing.yml"))
