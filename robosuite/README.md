# Master Thesis "Robotic Assembly of Structures with Unknown Objects Using Large Language Models"


Implementations on experiments for master thesis.


## Requirements

* Python 3.10 (also tested with 3.9)
* robosuite 1.5.1 (RoboticManipulation branch)
* openai 1.51.2


## Setup

This project is based on the [robosuite v1.5 with the passive viewer - RoboticManipulation/robosuite/tree/juhui branch](https://github.com/RoboticManipulation/robosuite/tree/juhui).

Clone the repository and set up the environment:

```bash
# Clone this repository
git clone https://gitlab.igg.uni-bonn.de/hrl_students/juhui_park/thesis-experiments.git
cd thesis-experiments

# Create and activate conda environment
conda create -n robosuite python=3.10
conda activate robosuite

# Install dependencies (see RoboticManipulation/robosuite/tree/juhui branch)
cd robosuite
pip install -e .
pip install -r requirements-extra.txt

# Install additional packages
pip install matplotlib \
            "llama-index==0.12.33" \
            llama-index-llms-mistralai \
            llama-index-llms-google-genai

cd robosuite
```

If you encounter the error `Install Error: Cannot initialize a EGL device display.`, try:

```bash
cd robosuite

# Fix for missing EGL / OpenGL support
# Reference: https://github.com/ARISE-Initiative/robosuite/issues/490#issuecomment-2169117010
conda install -c conda-forge gcc

pip install -e .
pip install -r requirements-extra.txt
```


## Test

### 1. Whole assembly framework from perception to execution

Set configurations by editing `configs/assembly_configs_integrated_move.json`. The main parameters to modify are:
```
{
    "env": {
        "env_name": str,    // task environment to test
        ...
    },
    "task": {
        "target_query_structure": str,    // query string to LLM
        ...
    },
    "llm": {
        "api_keys": {
            "openai": str,    // your own API key for LLMs
            ...
        },
        "use_llamaindex": false,    // whether to use LlamaIndex instead of OpenAI (false as default)
        ...
    },
    "which_img": str,    // image modality (see options below)
    "relation": {
        "use_primitives": bool,    // whether to use spatial relation primitives instead of center distance (true as default)
        ...
    }
}
```

For `env_name`, assembly tasks are implemented under `robosuite/environments/manipulation`. The available environments are categorized into demonstration tasks (for learning examples) and target tasks (for evaluation):
* Demonstration tasks:

| env_name | task | file |
|----------|------|------|
| Pyramid | Pyramid | `pyramid.py` |
| SmallTower | Small Tower | `small_tower.py` |
| Tower | Tower | `tower.py` |
| Square | Square | `square.py` |

* Target tasks:

| env_name | task | file |
|----------|------|------|
| BigPyramid | Big Pyramid | `big_pyramid.py` |
| BigTower | Big Tower | `big_tower.py` |
| TwinTowers | Towers (color) | `twin_towers.py` |
| House | House, House (tall) | `house.py` |
| BigHouse | Big House | `big_house.py` |
| BigHousePillars | Big House (color+horiz.), Big House (color+verti.) | `big_house_pillars.py` |


For `which_img`, specifying the image channels used for multimodal ICL, you can choose from the following options:
* `unpaired`: no image in demonstrations (only for center-distance relationships)
* `rgb`: RGB image only
* `rgb_d`: RGB and depth images separately
* `rgb_m`: RGB and mask images separately
* `rgb_d_m`: RGB, depth, and mask images separately
* `fused_rgbd`: fused 4-channel RGB-D
* `fused_rgbm`: RGB segmentation of building objects


For `llm/use_llamaindex`, you can choose which LLM API framework to use:
* `false` (default): import `StructurePlanner` from `structure_planner.py` (OpenAI)
* `true`: import `AssemblyLLM` from `structure_planner.py` (LlamaIndex)


> You can provide your API key in two ways:
> 
> 1. Save it in the JSON config file (e.g., `configs/assembly_configs_integrated_move.json`)
> 2. Set it as a system environment variable.
> 
> If you choose the environment variable option, use one of the following methods:
> ```bash
> # For the current terminal session only
> export OPENAI_API_KEY=your_api_key_here
> ```
> ```bash
> # For permanent use (bash users)
> echo 'export OPENAI_API_KEY=your_api_key_here' >> ~/.bashrc
> source ~/.bashrc
> ```


Run the following to test the framework in simulation:

```bash
export XDG_SESSION_TYPE=x11

cd demos
python demo_passive_viewer_integrated_move.py 
```


### 2. Sandwich task for evaluating generalization

Set configurations by editing `configs/assembly_configs_sandwich.json`:
```
{
    "task": {
        "target_query_structure": str,    // query string to LLM, e.g., "sandwich", "vegetarian sandwich", and "sandwich for milk allergy"
        ...
    },
    "llm": {
        "api_keys": {
            "openai": str,    // your own API key for LLMs
            ...
        },
        "use_llamaindex": false,    // whether to use LlamaIndex instead of OpenAI (false as default)
    }
}
```

Run the following to test the Sandwich task:

```bash
cd demos
python plan_sandwich_assembly.py 
```
