# Agile Robot Control (ARC) Bridge
Software bridge of a nominal robot controller to Mujoco simulator via LCM communication protocol.

## TODOs
- [x] Support CI via GitHub action
- [x] Automate installation steps on Windows
- [x] Verify installation steps on macOS
- [ ] Add a flag to turn state estimator (if applicable) on/off
- [ ] Support PlotJuggler on the fly

## Quick Start
1. Make sure you have java installed. Type `javac` to verify.
1. Clone this repo using **SSH**.
    ```sh
    git clone git@github.com:ARCaD-Lab-UM/arc-bridge.git
    ```
1. Create a conda environment.
    ```sh
    conda env create -f environment.yml
    conda activate arcpy
    ```
1. Generate LCM types.
    ```sh
    bash gen_lcm_types.sh # Ubuntu bash
    gen_lcm_types_win.cmd # Windows cmd
    ```
    :warning: Redo this step if any LCM types are changed.
1. Install `arc-bridge` as a Python module in editable and compatible mode.
    ```sh
    pip install -e . --no-deps --config-settings editable_mode=compat
    ```
1. Launch it in any command line with `arcpy` activated.
    ```sh
    arc-bridge
    ```
    Use `--help` to find other launching options.
1. Check the communication status and visualize data.
    ```sh
    arc-lcm-spy
    ```

## Supported Robot & Controller
:point_down: Click to find the corresponding controller
- [x] [MUPS Hopper](https://github.com/ARCaD-Lab-UM/mups-controller)
- [x] [LimX Tron1 Pointfoot](https://github.com/ARCaD-Lab-UM/tron1-model-based-controller/blob/main/point_foot/MAIN_PF_LCM.m)
- [x] [LimX Tron1 Linefoot](https://github.com/ARCaD-Lab-UM/tron1-model-based-controller/blob/main/line_foot/MAIN_LF_LCM.m)
- [x] [Toy Biped Linefoot](https://github.com/ARCaD-Lab-UM/TrainingWheel/blob/main/control_Cassie/MAIN_cassie_LCM.m)
- [x] [Toy Biped Pointfoot](https://github.com/ARCaD-Lab-UM/TrainingWheel/blob/main/control_tron1/MAIN_tron1_LCM.m)

## Dependencies
- [LCM](https://github.com/lcm-proj/lcm)
- [Mujoco](https://github.com/google-deepmind/mujoco)
- Java (LCM type generation and visualization)

## Troubleshooting

<details>
    <summary>  
        <b> For macOS users </b>
    </summary>

Use `mjpython` instead of `python` to launch the bridge.
</details>

<details>
    <summary>  
        <b> LCM messages not found in MATLAB </b>
    </summary>

Restart MATLAB once after generating LCM types.
</details>

<details>
    <summary>  
        <b> GLFW Error </b>
    </summary>

```sh
GLFWError: (65542) b'GLX: No GLXFBConfigs returned'
GLFWError: (65545) b'GLX: Failed to find a suitable GLXFBConfig'
ERROR: could not create window
```
Set NVIDIA GPU as primary renderer (for systems with NVIDIA GPUs)
```
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```
</details>

<details>
    <summary>  
        <b> limits.h: No such file or directory </b>
    </summary>

When compiling LCM, disable unit tests.
```sh
cmake .. -DLCM_ENABLE_EXAMPLES=OFF -DLCM_ENABLE_TESTS=OFF
```
</details>

<details>
    <summary>
        <b> Unable to install LimX SDK in arcpy </b>
    </summary>

Downgrade `mujoco` to 3.2.2 and `numpy` to 1.21.6 manually.
</details>
