# Agile Robot Control (ARC) Bridge
Software bridge of a nominal robot controller to Mujoco simulator via LCM communication protocol.

## Dependencies
- [LCM](https://github.com/lcm-proj/lcm)
- [Mujoco](https://github.com/google-deepmind/mujoco)
- Java (LCM type generation and visualization)

## Quick Start
1. Make sure you have java installed. Type `javac` to verify.
2. Clone this repo using **SSH**.
    ```sh
    git clone git@github.com:ARCaD-Lab-UM/arc-bridge.git
    ```
3. Build and **install** LCM from source by following the [official instructions](https://lcm-proj.github.io/lcm/content/build-instructions.html).
4. Check [this doc](https://lcm-proj.github.io/lcm/content/java-notes.ml#finding-lcm-jar) to find the compiled `lcm.jar` and copy it to `<lcm_types>` folder.
5. Generate LCM types.
    ```sh
    gen_lcm_types.sh
    ```
    :warning: Redo this step if any LCM types are changed (may from new commits).
6. Create the conda environment and install `arc-bridge`.
    ```sh
    conda env create -f environment.yml
    conda activate arcpy
    pip install -e . --no-deps --config-settings editable_mode=compat
    ```
7. Launch it in command line and follow the prompt to select a robot type.
    ```sh
    arc-bridge
    ```
    Use `--help` to find other launching options.
8. Spy the communication rate and plot data.
    ```sh
    arc-lcm-spy
    ```

## Supported Robot & Controller
:point_down: Click to find the corresponding controller
- [x] [PEA Hopper](https://github.com/ARCaD-Lab-UM/hopper-kd-mpc/blob/main/HopperMain.m)
- [x] [LimX Tron1 Pointfoot](https://github.com/ARCaD-Lab-UM/tron1-model-based-controller/blob/main/point_foot/MAIN_PF_LCM.m)
- [x] [LimX Tron1 Linefoot](https://github.com/ARCaD-Lab-UM/tron1-model-based-controller/blob/main/line_foot/MAIN_LF_LCM.m)
- [x] [Linefoot Biped](https://github.com/ARCaD-Lab-UM/TrainingWheel/blob/main/ex_Cassie/MAIN_cassie_LCM.m)
- [x] [Pointfoot Biped](https://github.com/ARCaD-Lab-UM/TrainingWheel/blob/main/ex_tron1/MAIN_tron1_LCM.m)
<!-- - [x] 2 Link Arm -->


## Troubleshooting

<details>
    <summary>  
        <b> For macOS or Windows users </b>
    </summary>

macOS: Use `mjpython` instead of `python` to launch the bridge.
<br>
Windows: 
1. Change the `__init__.py` of installed LCM package in `site-packages` based on this [PR](https://github.com/lcm-proj/lcm/pull/581)
2. Install `glib` in conda.
3. Run `gen_lcm_types_win.cmd` to generate LCM types.
4. Install `arc-bridge` as in step 6.
</details>

<details>
    <summary>  
        <b> LCM messages not found in MATLAB </b>
    </summary>

Restart MATLAB once after generating LCM types.
</details>


<details>
    <summary>  
        <b> Permission denied when initializing LCM as a submodule </b>
    </summary>

Use **SSH** option to clone this repo.
```sh
git clone --recursive git@github.com:ARCaD-Lab-UM/agile-robot-control-bridge.git
```
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
