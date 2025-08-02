import os
import subprocess
import sys
from pathlib import Path


def main():
    # Get the package installation directory
    package_dir = Path(__file__).parent
    jar_path = package_dir / "lcm_msgs" / "lcm_msgs.jar"

    if not jar_path.exists():
        print(f"Error: LCM messages JAR not found at {jar_path}")
        print("Please ensure the package was installed correctly")
        sys.exit(1)

    # Set classpath and launch lcm-spy
    env = os.environ.copy()
    env['CLASSPATH'] = str(jar_path)

    try:
        subprocess.run(['lcm-spy'], env=env)
    except FileNotFoundError:
        print("Error: lcm-spy not found. Please install LCM tools.")
        sys.exit(1)


if __name__ == "__main__":
    main()
