@REM Generate LCM types for Windows
@echo off
for %%f in ("lcm_types\*.lcm") do (
    lcm-gen -p "%%f" --ppath arc_bridge
)

@echo on
echo LCM types generation completed.
