import pytest

import torch


torch.cuda.is_available = lambda: False  # type: ignore[assignment]
torch.cuda.current_device = lambda: 0  # type: ignore[assignment]
torch.cuda.get_device_name = lambda device: "stub-device"  # type: ignore[assignment]
torch.cuda.memory_stats = lambda dev: {"reserved_bytes.all.current": 0}  # type: ignore[assignment]
torch.cuda.mem_get_info = lambda dev: (0, 0)  # type: ignore[assignment]

import nodes
from comfy_execution.graph import DynamicPrompt, ExecutionList
from execution import extract_node_groups


class _DummyOutputCache:
    def __init__(self):
        self._storage = {}

    def get(self, node_id):
        return self._storage.get(node_id)


class BatchTestInput:
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @staticmethod
    def execute():
        return (1,)


class BatchTestOutput:
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT",), "y": ("INT",)}}

    @staticmethod
    def execute(x, y):
        return ()


@pytest.fixture(autouse=True)
def register_batch_test_nodes():
    mapping = {
        "BatchTestInput": BatchTestInput,
        "BatchTestOutput": BatchTestOutput,
    }
    originals = {name: nodes.NODE_CLASS_MAPPINGS.get(name) for name in mapping}
    try:
        nodes.NODE_CLASS_MAPPINGS.update(mapping)
        yield
    finally:
        for name, original in originals.items():
            if original is None:
                nodes.NODE_CLASS_MAPPINGS.pop(name, None)
            else:
                nodes.NODE_CLASS_MAPPINGS[name] = original


def _build_prompt():
    return {
        "input1": {"class_type": "BatchTestInput", "inputs": {}},
        "input2": {"class_type": "BatchTestInput", "inputs": {}},
        "output": {
            "class_type": "BatchTestOutput",
            "inputs": {
                "x": ["input1", 0],
                "y": ["input2", 0],
            },
        },
    }


def test_extract_node_groups_from_extra_data():
    prompt = _build_prompt()
    extra_data = {
        "node_groups": {
            "batch-1": {
                "variant": "batch",
                "nodes": ["input1", "input2"],
            }
        }
    }

    config = extract_node_groups(prompt, extra_data)

    assert "batch-1" in config
    assert config["batch-1"]["variant"] == "batch"
    assert config["batch-1"]["nodes"] == {"input1", "input2"}


def test_extract_node_groups_from_node_metadata():
    prompt = _build_prompt()
    prompt["input1"]["group"] = "batch-2"
    prompt["input1"]["group_variant"] = "batch"
    prompt["input2"]["group"] = {"id": "batch-2"}

    config = extract_node_groups(prompt, {})

    assert "batch-2" in config
    assert config["batch-2"]["variant"] == "batch"
    assert config["batch-2"]["nodes"] == {"input1", "input2"}


@pytest.mark.asyncio
async def test_execution_list_batches_nodes_once_ready():
    prompt = _build_prompt()
    extra_data = {
        "node_groups": {
            "batch-group": {
                "variant": "batch",
                "nodes": ["input1", "input2"],
            }
        }
    }
    dynamic_prompt = DynamicPrompt(prompt)
    cache = _DummyOutputCache()
    group_config = extract_node_groups(prompt, extra_data)
    execution_list = ExecutionList(dynamic_prompt, cache, group_config)

    execution_list.add_node("output")

    staged, error, ex = await execution_list.stage_node_execution()

    assert error is None and ex is None
    assert isinstance(staged, list)
    assert set(staged) == {"input1", "input2"}

    execution_list.unstage_node_execution()
