# MolmoBot-DROID

`MolmoBotDroidJointposClient` talks to a MolmoBot-DROID policy server over the
openpi-compatible msgpack/websocket protocol. Connection is lazy: the client
does not open a socket until the first inference call, so it can be
constructed (and unit-tested) with no server running.

See the [policies README](../README.md) for the shared client architecture.

## Operational constraint: `--num-envs 1` only

The MolmoBot server keeps per-session state (an internal 16-step action
buffer advanced one action per request). Running `policies/molmobot/run.py`
with `--num-envs N>1` multiplexes N interleaved env streams into that single
buffer and produces corrupted actions. Until the server is patched to return
the full chunk per request (see the study runbook), run MolmoBot evaluations
with `--num-envs 1`.

**Patched-server mode:** launch the server from the `serve/full-chunk`
branch of the `Chung-I/MolmoBot` fork with `--serve-full-chunk`; serving
is then stateless, so run with `--allow-multi-env --num-envs 16`. The
client detects the mode from the response and adopts the server's
`execute_horizon` (8) and per-step `relative_max_joint_delta` clamp
automatically.

## Install the client

`openpi_client` is required. If it is not already installed in the RoboLab
venv:
```bash
cd robolab
uv pip install -e ../openpi/packages/openpi-client
```

## Start the policy server

The server lives in the separate `MolmoBot` repo (`olmo/eval/websocket_server.py`
+ `launch_scripts/serve_molmo.py`) and is NOT installed into the RoboLab venv —
run it in its own environment, e.g. on cml30:

```bash
PYTHONPATH=. python launch_scripts/serve_molmo.py \
    --hf-repo allenai/MolmoBot-DROID \
    --action-type joint_pos
```

This serves on port 8000 by default (`WebsocketPolicyServer(..., port=8000)`
in `olmo/eval/websocket_server.py`).

## Connect from RoboLab

cml30 is directly reachable from the RTX 5090 / other eval hosts, so no SSH
tunnel is needed — connect straight to `<cml30-host>:8000`:

```python
from policies.molmobot.client import MolmoBotDroidJointposClient

client = MolmoBotDroidJointposClient(remote_host="cml30.csie.ntu.edu.tw", remote_port=8000)
```

(Swap in whatever host actually runs `serve_molmo.py`; `localhost:8000` if
it's running on the same box as the eval loop.)

## Wire-format findings (Task 6 step 1)

Resolved by reading `~/Codes/MolmoBot` (see `client.py`'s module docstring
for full file:line evidence):

1. **Protocol**: MolmoBot's `websocket_server.py` is an explicit fork of
   openpi's `websocket_policy_server.py` (msgpack-numpy metadata handshake,
   then pack/send/recv/unpack request-response loop) — so this client reuses
   `openpi_client.websocket_client_policy.WebsocketClientPolicy` directly
   instead of vendoring a protocol implementation.
2. **Chunk shape → `DEFAULT_HORIZON = 1`**: `RealRobotVLAPolicy.get_action`
   (`configure_real_robot.py:160`) returns exactly one `{"arm": (7,),
   "gripper": (1,)}` action dict per websocket round trip. The server
   internally predicts a 16-step chunk and re-queries its model only every
   8 calls, but that buffering never crosses the wire — the client must
   requery every environment step.
3. **Gripper convention**: MolmoBot's default `clamp_gripper=True` produces
   values on a 0-255 scale (`>128 -> 255` = closed, else `0` = open;
   documented at `synthmanip_grasp_sampling.py:53`). This is the *same*
   polarity as RoboLab's 0=open/1=closed convention
   (`robolab/robots/droid.py:228`), just a different scale — `_unpack_response`
   divides by 255, no inversion.
