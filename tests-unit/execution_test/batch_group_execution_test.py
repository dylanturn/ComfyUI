import asyncio
import time

import pytest

import comfy.cli_args as cli_args

cli_args.args.cpu = True

import nodes
from execution import PromptExecutor


class DummyServer:
    def __init__(self):
        self.client_id = None
        self.last_node_id = None

    def send_sync(self, *args, **kwargs):
        pass

    def queue_updated(self):
        pass


class BatchBarrier:
    start_counter: int = 0
    results: list[tuple] = []

    @classmethod
    def reset(cls):
        cls.start_counter = 0
        cls.results = []

    @classmethod
    async def wait_for_all(cls, label: str):
        cls.results.append(("start", label, time.perf_counter()))
        cls.start_counter += 1
        await asyncio.sleep(0.05)
        cls.results.append(("end", label, time.perf_counter()))
        return (label,)


class BatchNodeA:
    CATEGORY = "tests"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    async def execute(cls):
        return await BatchBarrier.wait_for_all("A")


class BatchNodeB:
    CATEGORY = "tests"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    async def execute(cls):
        return await BatchBarrier.wait_for_all("B")


class BatchOutputNode:
    CATEGORY = "tests"
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first": ("STRING", {}),
                "second": ("STRING", {}),
            }
        }

    @classmethod
    def execute(cls, first, second):
        BatchBarrier.results.append(("output", first, second))
        return ()


class AsyncSourceFast:
    CATEGORY = "tests"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    async def execute(cls):
        BatchBarrier.results.append(("source_start", "fast", time.perf_counter()))
        await asyncio.sleep(0.01)
        BatchBarrier.results.append(("source_end", "fast", time.perf_counter()))
        return ("fast",)


class AsyncSourceSlow:
    CATEGORY = "tests"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    async def execute(cls):
        BatchBarrier.results.append(("source_start", "slow", time.perf_counter()))
        await asyncio.sleep(0.05)
        BatchBarrier.results.append(("source_end", "slow", time.perf_counter()))
        return ("slow",)


class DependentBatchNodeA:
    CATEGORY = "tests"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {})}}

    @classmethod
    async def execute(cls, value):
        return await BatchBarrier.wait_for_all("A_dep")


class DependentBatchNodeB:
    CATEGORY = "tests"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {})}}

    @classmethod
    async def execute(cls, value):
        return await BatchBarrier.wait_for_all("B_dep")


@pytest.fixture(autouse=True)
def register_batch_nodes():
    classes = {
        "BatchNodeA": BatchNodeA,
        "BatchNodeB": BatchNodeB,
        "BatchOutputNode": BatchOutputNode,
        "AsyncSourceFast": AsyncSourceFast,
        "AsyncSourceSlow": AsyncSourceSlow,
        "DependentBatchNodeA": DependentBatchNodeA,
        "DependentBatchNodeB": DependentBatchNodeB,
    }
    previous = {}
    for name, cls in classes.items():
        previous[name] = nodes.NODE_CLASS_MAPPINGS.get(name)
        nodes.NODE_CLASS_MAPPINGS[name] = cls
    try:
        yield
    finally:
        for name, original in previous.items():
            if original is None:
                nodes.NODE_CLASS_MAPPINGS.pop(name, None)
            else:
                nodes.NODE_CLASS_MAPPINGS[name] = original


@pytest.mark.asyncio
async def test_batch_group_executes_nodes_concurrently():
    BatchBarrier.reset()
    server = DummyServer()
    executor = PromptExecutor(server)

    prompt = {
        "1": {"class_type": "BatchNodeA", "inputs": {}},
        "2": {"class_type": "BatchNodeB", "inputs": {}},
        "3": {
            "class_type": "BatchOutputNode",
            "inputs": {
                "first": ["1", 0],
                "second": ["2", 0],
            },
        },
    }

    extra_data = {
        "node_execution_groups": {
            "group": {
                "variant": "batch",
                "nodes": ["1", "2"],
            }
        }
    }

    await executor.execute_async(prompt, "test_prompt", extra_data, execute_outputs=["3"])

    assert executor.success

    start_entries = [entry for entry in BatchBarrier.results if entry[0] == "start"]
    end_entries = [entry for entry in BatchBarrier.results if entry[0] == "end"]
    assert len(start_entries) == 2
    assert len(end_entries) == 2

    output_index = next(i for i, entry in enumerate(BatchBarrier.results) if entry[0] == "output")
    assert end_entries[-1][2] >= end_entries[0][2]

    first_end_index = next(i for i, entry in enumerate(BatchBarrier.results) if entry[0] == "end")
    start_indices = [i for i, entry in enumerate(BatchBarrier.results) if entry[0] == "start"]
    assert start_indices[-1] < first_end_index
    end_indices = [i for i, entry in enumerate(BatchBarrier.results) if entry[0] == "end"]
    assert all(i < output_index for i in end_indices)


@pytest.mark.asyncio
async def test_batch_group_waits_for_all_dependencies_before_start():
    BatchBarrier.reset()
    server = DummyServer()
    executor = PromptExecutor(server)

    prompt = {
        "10": {"class_type": "AsyncSourceFast", "inputs": {}},
        "11": {"class_type": "AsyncSourceSlow", "inputs": {}},
        "20": {
            "class_type": "DependentBatchNodeA",
            "inputs": {
                "value": ["10", 0],
            },
        },
        "21": {
            "class_type": "DependentBatchNodeB",
            "inputs": {
                "value": ["11", 0],
            },
        },
        "30": {
            "class_type": "BatchOutputNode",
            "inputs": {
                "first": ["20", 0],
                "second": ["21", 0],
            },
        },
    }

    extra_data = {
        "node_execution_groups": {
            "group": {
                "variant": "batch",
                "nodes": ["20", "21"],
            }
        }
    }

    await executor.execute_async(prompt, "test_prompt", extra_data, execute_outputs=["30"])

    assert executor.success

    source_end_times = [entry[2] for entry in BatchBarrier.results if entry[0] == "source_end"]
    assert len(source_end_times) == 2

    batch_start_entries = [
        entry for entry in BatchBarrier.results if entry[0] == "start" and entry[1] in {"A_dep", "B_dep"}
    ]
    assert len(batch_start_entries) == 2

    first_batch_start = min(entry[2] for entry in batch_start_entries)
    assert first_batch_start >= max(source_end_times)

    batch_end_entries = [
        entry for entry in BatchBarrier.results if entry[0] == "end" and entry[1] in {"A_dep", "B_dep"}
    ]
    assert len(batch_end_entries) == 2

    start_indices = [
        i for i, entry in enumerate(BatchBarrier.results) if entry[0] == "start" and entry[1] in {"A_dep", "B_dep"}
    ]
    first_end_index = next(
        i for i, entry in enumerate(BatchBarrier.results) if entry[0] == "end" and entry[1] in {"A_dep", "B_dep"}
    )
    assert start_indices[-1] < first_end_index
