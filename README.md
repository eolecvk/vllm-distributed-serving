# Setup for multi-node llama-3

Hop onto cluster
```bash
ssh -i <key_file> -F ~/.ssh/config.d/config.ml-512 ml-512-node-061
ssh -i <key_file> -F ~/.ssh/config.d/config.ml-512 ml-512-node-062
ssh -i <key_file> -F ~/.ssh/config.d/config.ml-512 ml-512-node-063
ssh -i <key_file> -F ~/.ssh/config.d/config.ml-512 ml-512-node-064
```

## Setup ray runtime on multi-node GPU cluster

[documentation reference](https://vllm--6529.org.readthedocs.build/en/6529/serving/distributed_serving.html#multi-node-inference-and-serving)

Install ray on each node
```bash
pip install ray
```

Pick node as head node, and run
```bash
ray start --head
# Local node IP: 172.26.135.124
```
will return message like `To add another node to this Ray cluster, run ray start --address='xxx.xxx.xxx.xxx:6379'`

On other nodes, run
```bash
ray start --address='xxx.xxx.xxx.xxx:6379'
# ray start --address='172.26.135.124:6379'
```

Sanity check

To make sure all nodes joined successfully, run from any node:
```bash
ray status
```

Check GPU-GPU communication

Create `test.py` on each node
```python
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
data = torch.FloatTensor([1,] * 128).to(f"cuda:{local_rank}")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"

gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("sanity check is successful!")
```

Run `test.py` on each node:
```
RANK=<0 for master and 1+ for workers>
MASTER_ADDR=172.26.135.124:1234
torchrun \
--nproc_per_node=8 \
--nnodes=4 \
--node_rank=$RANK \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR \
/home/ubuntu/ml-Illinois/eole/test.py
```

<!-- Uploading "image.png"... -->





## Multi-node Inference and Serving

source: https://vllm--6529.org.readthedocs.build/en/6529/serving/distributed_serving.html

```bash
pip install vllm jsonschema
```

```bash
huggingface-cli login
```

Run `vllm serve` with `pipeline-parallel` on master
```bash
vllm GPT2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

**Ray cluster crash error**

Log on master:
```
2024-07-19 11:36:24,756	INFO worker.py:1788 -- Connected to Ray cluster.
[2024-07-19 11:36:53,765 E 1446555 1446656] core_worker_process.cc:217: Failed to get the system config from raylet because it is dead. Worker will terminate. Status: RpcError: RPC Error message: failed to connect to all addresses; last error: UNKNOWN: ipv4:172.26.135.124:41677: connection attempt timed out before receiving SETTINGS frame; RPC Error details:  .Please see `raylet.out` for more details.
```

`tail -n 100 /tmp/ray/session_latest/logs` on worker:
```
[2024-07-19 11:36:24,756 I 1284440 1284440] raylet: Starting Raylet with ID
[2024-07-19 11:36:41,302 W 1284440 1284440] (raylet) memory_monitor.cc:220:  file not found: /proc/meminfo
[2024-07-19 11:36:41,302 W 1284440 1284440] (raylet) memory_monitor.cc:81: Unable to capture node memory. Monitor will not be able to detect memory usage above threshold.
[2024-07-19 11:36:44,553 W 1284440 1284440] (raylet) memory_monitor.cc:197: Got negative used memory for cgroup -1, setting it to zero
[2024-07-19 11:36:46,303 W 1284440 1284440] (raylet) memory_monitor.cc:220:  file not found: /proc/meminfo
[2024-07-19 11:36:46,303 W 1284440 1284440] (raylet) memory_monitor.cc:81: Unable to capture node memory. Monitor will not be able to detect memory usage above threshold.
[2024-07-19 11:36:49,554 W 1284440 1284440] (raylet) memory_monitor.cc:197: Got negative used memory for cgroup -1, setting it to zero
[2024-07-19 11:36:51,304 W 1284440 1284440] (raylet) memory_monitor.cc:220:  file not found: /proc/meminfo
[2024-07-19 11:36:51,304 W 1284440 1284440] (raylet) memory_monitor.cc:81: Unable to capture node memory. Monitor will not be able to detect memory usage above threshold.
[2024-07-19 11:36:54,554 W 1284440 1284440] (raylet) memory_monitor.cc:197: Got negative used memory for cgroup -1, setting it to zero
[2024-07-19 11:36:56,305 W 1284440 1284440] (raylet) memory_monitor.cc:220:  file not found: /proc/meminfo
[2024-07-19 11:36:56,305 W 1284440 1284440] (raylet) memory_monitor.cc:81: Unable to capture node memory. Monitor will not be able to detect memory usage above threshold.
[2024-07-19 11:36:59,555 W 1284440 1284440] (raylet) memory_monitor.cc:197: Got negative used memory for cgroup -1, setting it to zero
[2024-07-19 11:37:01,306 W 1284440 1284440] (raylet) memory_monitor.cc:220:  file not found: /proc/meminfo
[2024-07-19 11:37:01,306 W 1284440 1284440] (raylet) memory_monitor.cc:81: Unable to capture node memory. Monitor will not be able to detect memory usage above threshold.
[2024-07-19 11:37:04,556 W 1284440 1284440] (raylet) memory_monitor.cc:197: Got negative used memory for cgroup -1, setting it to zero
[2024-07-19 11:37:06,306 W 1284440 1284440] (raylet) memory_monitor.cc:220:  file not found: /proc/meminfo
[2024-07-19 11:37:06,306 W 1284440 1284440] (raylet) memory_monitor.cc:81: Unable to capture node memory. Monitor will not be able to detect memory usage above threshold.
[2024-07-19 11:37:09,557 W 1284440 1284440] (raylet) memory_monitor.cc:197: Got negative used memory for cgroup -1, setting it to zero
[2024-07-19 11:37:11,307 W 1284440 1284440] (raylet) memory_monitor.cc:220:  file not found: /proc/meminfo
[2024-07-19 11:37:11,307 W 1284440 1284440] (raylet) memory_monitor.cc:81: Unable to capture node memory. Monitor will not be able to detect memory usage above threshold.
```
