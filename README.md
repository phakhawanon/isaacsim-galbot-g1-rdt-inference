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
config/
  rdt_server.yaml          # selects the RDT-1B checkpoint/model for rdt_server.py
  precompute_lang_embed.yaml # selects the task instruction/T5 checkpoint for precompute_lang_embed.py
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

1. Precompute the task's language embedding (in the `rdt` conda env, once --
   only needs to be re-run if the task instruction changes). Set the actual
   task instruction in `config/precompute_lang_embed.yaml` first -- see
   [Selecting the task instruction](#selecting-the-task-instruction) below --
   then run:

   ```bash
   conda activate rdt
   cd ../rdt-1b-galbot && python ../inference_galbot_golf/scripts/precompute_lang_embed.py
   ```

   This writes `src/rdt_inference/lang_embed.pt`, which `rdt_server.py` loads
   at startup -- the server will fail to start without it.

2. (Optional) Select which RDT-1B checkpoint/model to serve -- see
   [Selecting the RDT-1B model](#selecting-the-rdt-1b-model) below. The
   server ships with a default pointing at `checkpoint-70000`.

3. Start the RDT-1B inference server (in the `rdt` conda env):

   ```bash
   conda activate rdt
   ./scripts/start_rdt_server.sh
   ```

4. In a separate terminal, launch the simulator (in the `isaaclab` conda env):

   ```bash
   conda activate isaaclab
   ./scripts/run_main.sh
   ```

   `main.py` connects to the RDT server over TCP (`localhost:8765` by
   default, override with `RDT_SERVER_ADDR=host:port`), streams camera +
   proprioceptive observations to it each control tick, and executes the
   returned action chunks on the simulated Galbot.

### Selecting the RDT-1B model

`rdt_server.py` pulls variables which specify the location of RDT-1B model from `config/rdt_server.yaml`:

```yaml
rdt_root: "../rdt-1b-galbot"
checkpoint: "checkpoints/rdt-1b-finetune/checkpoint-70000"
vision_encoder: "google/siglip-so400m-patch14-384"
control_frequency: 30
```

- `rdt_root`: path to your `rdt-1b-galbot` checkout. Relative paths are
  resolved relative to this repo's root.
- `checkpoint`: checkpoint directory to load, relative to `rdt_root` (or an
  absolute path).
- `vision_encoder`: HuggingFace model name/path for the vision encoder the
  checkpoint above was fine-tuned with.
- `control_frequency`: control frequency (Hz) the checkpoint was fine-tuned at.

To switch models, either edit `config/rdt_server.yaml` in place, or keep
multiple config files (e.g. `config/rdt_server.putting.yaml`,
`config/rdt_server.driving.yaml`) and select one without touching the
default:

```bash
RDT_SERVER_CONFIG=/path/to/other.yaml ./scripts/start_rdt_server.sh
```

### Selecting the task instruction

`precompute_lang_embed.py` reads the task instruction and T5 encoder model from
`config/precompute_lang_embed.yaml`:

```yaml
rdt_root: "../rdt-1b-galbot"
gpu: 0
t5_model: "t5-v1_1-xxl"
task_name: "Pouring"
instruction: "Pour the water into the cup"
offload_dir: null
save_path: "src/rdt_inference/lang_embed.pt"
```

- `rdt_root`: path to your `rdt-1b-galbot` checkout, same convention as
  `config/rdt_server.yaml`.
- `gpu`: CUDA device index to run the T5 encoder on.
- `t5_model`: local T5-xxl checkpoint directory, relative to `rdt_root` (or
  an absolute path).
- `task_name` / `instruction`: the label and actual instruction text the
  RDT-1B policy should condition on -- edit `instruction` before running.
- `offload_dir`: set to an existing directory to enable CPU offloading if
  your GPU has less than 24GB VRAM.
- `save_path`: where the resulting `lang_embed.pt` is written, relative to
  this repo's root. Leave this pointed at `src/rdt_inference/lang_embed.pt`
  unless you also change where `rdt_server.py` looks for it.

As with the server config, point the script at a different file with:

```bash
PRECOMPUTE_LANG_EMBED_CONFIG=/path/to/other.yaml python scripts/precompute_lang_embed.py
```

### Adding scenes in Isaac Sim

User can add customized scenes for the simulation is Isaac Sim by creating new .py files inside
`src/isaacsim_interface/scenes/`. The new .py file should at least contain 

   ```python 
   def design_scene() -> tuple[Articulation, Camera, Camera, Camera]:
   ```

which is the logic behind scene construction. (See `src/isaacsim_interface/scenes/default.py` for example.)
Do not forget to include the necessary imports to accomodate the design_scene() function.

To select the scene for simulation, edit `src/isaacsim_inference/main.py` at line 58 to

   ```python
   from isaacsim_interface.scenes.<scene_file_name> import design_scene
   ```

### Deploying on a real robot

`rdt_server.py` only speaks the plain TCP wire protocol defined in
`src/rdt_inference/rdt_ipc.py` (pickled messages, length-prefixed) -- it has
no dependency on Isaac Sim. A real robot's control process can act as a
client the same way `main.py` does:

1. Precompute the language embedding and start the RDT-1B inference server
   as above, on a host reachable from the robot's control box (it binds
   `0.0.0.0` by default).
2. From the robot's control process, connect to that `host:port` and use
   `rdt_inference.rdt_ipc` (`parse_addr`, `send_msg`, `recv_msg`,
   `encode_image`) to send `{"proprio": ..., "images": {...}}` observations
   and receive `{"action_chunk": [...]}` in response, following the same
   request/response shape `main.py` uses in
   `src/isaacsim_inference/main.py`.

The wire protocol module is deliberately pure-stdlib + PIL only, so it can
be imported unmodified regardless of which environment the robot's control
process runs in.
