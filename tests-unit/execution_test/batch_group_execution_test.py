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


@pytest.fixture(autouse=True)
def register_batch_nodes():
    classes = {
        "BatchNodeA": BatchNodeA,
        "BatchNodeB": BatchNodeB,
        "BatchOutputNode": BatchOutputNode,
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
