# isaacsim-galbot-g1-rdt-inference

IsaacLab + RDT-1B inference stack for the Galbot G1 Foxtrot task. The Galbot
articulation is simulated in Isaac Sim and driven by actions produced by an
RDT-1B policy running as a separate, persistent inference server. The two
halves talk to each other over a plain TCP socket, so the `isaaclab` and
`rdt` conda environments (which pin conflicting torch/CUDA versions) never
need to coexist in one process -- and the same server can later be pointed
at by a real robot's control process instead of the simulator.

## Repository layout

```
src/
  isaacsim_interface/   # Isaac Sim scene setup + robot I/O helpers
  rdt_inference/         # RDT-1B inference server + client/server wire protocol
  isaacsim_inference/    # simulation entry point (main.py)
scripts/
  run_main.sh             # launches main.py inside Isaac Sim
  start_rdt_server.sh      # launches the RDT-1B inference server
  precompute_lang_embed.py # precomputes the task's language embedding
```

## Installation

1. Clone this repository next to your `rdt-1b-galbot` checkout (the RDT
   server imports it as a sibling directory):

   ```bash
   git clone https://github.com/phakhawanon/isaacsim-galbot-g1-rdt-inference
   ```

2. Set up the `isaaclab` conda environment following your local IsaacLab
   installation instructions.

3. Set up the `rdt` conda environment following the installation instructions at
   [https://github.com/thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer).

## Usage

### Deploying in Isaac Sim

Since Galbot G1 Foxtrot is not a default robotic articulation provided by IsaacLab,
user must install the robot config by themseleves.
**The installation steps for Galbot G1 Foxtrot config will be posted here if finished**

Inside the root of this repo.

1. Start the RDT-1B inference server (in the `rdt` conda env):

   ```bash
   conda activate rdt
   ./scripts/start_rdt_server.sh
   ```

2. In a separate terminal, launch the simulator (in the `isaaclab` conda env):

   ```bash
   conda activate isaaclab
   ./scripts/run_main.sh
   ```

   `main.py` connects to the RDT server over TCP (`localhost:8765` by
   default, override with `RDT_SERVER_ADDR=host:port`), streams camera +
   proprioceptive observations to it each control tick, and executes the
   returned action chunks on the simulated Galbot.

### Adding scenes in Isaac Sim

User can add customized scenes for the simulation is Isaac Sim by creating new .py files inside
`src/isaacsim_interface/scenes/`. The new .py file should at least contain 

   ```python 
   def design_scene() -> tuple[Articulation, Camera, Camera, Camera]:
   ```

which is the logic behind scene construction. (See `src/isaacsim_interface/scenes/default.py` for example.)
Do not forget to include the necessary imports to accomodate the design_scene() function.

### Deploying on a real robot

`rdt_server.py` only speaks the plain TCP wire protocol defined in
`src/rdt_inference/rdt_ipc.py` (pickled messages, length-prefixed) -- it has
no dependency on Isaac Sim. A real robot's control process can act as a
client the same way `main.py` does:

1. Start the RDT-1B inference server as above, on a host reachable from the
   robot's control box (it binds `0.0.0.0` by default).
2. From the robot's control process, connect to that `host:port` and use
   `rdt_inference.rdt_ipc` (`parse_addr`, `send_msg`, `recv_msg`,
   `encode_image`) to send `{"proprio": ..., "images": {...}}` observations
   and receive `{"action_chunk": [...]}` in response, following the same
   request/response shape `main.py` uses in
   `src/isaacsim_inference/main.py`.

The wire protocol module is deliberately pure-stdlib + PIL only, so it can
be imported unmodified regardless of which environment the robot's control
process runs in.
