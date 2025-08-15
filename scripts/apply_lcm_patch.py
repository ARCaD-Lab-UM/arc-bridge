#!/usr/bin/env python3
"""
Apply LCM Windows compatibility patch from GitHub PR #581
Downloads the fixed function from the official LCM repository and patches the local installation.
"""
import re
import sys
import os
import urllib.request
import urllib.error


def apply_lcm_windows_patch(lcm_init_file):
    """Apply the Windows compatibility patch to LCM __init__.py file."""
    
    # GitHub URL for the fixed version
    github_url = "https://raw.githubusercontent.com/lcm-proj/lcm/58aa6d2d4bf977da1f8fdfc78690c80dc8253e70/lcm-python/lcm/__init__.py"
    
    print(f"Patching LCM file: {lcm_init_file}")
    
    try:
        # Read the original file
        with open(lcm_init_file, 'r') as f:
            original_content = f.read()
        
        # Download the fixed version from GitHub
        print("Downloading fixed version from GitHub...")
        with urllib.request.urlopen(github_url) as response:
            fixed_content = response.read().decode('utf-8')
        
        # Extract the fixed implementation (the specific lines from the PR)
        pattern = r'if platform\.system\(\) == \'Windows\':\s*.*?raise FileNotFoundError\(f"No executable found for {name} in {LCM_BIN_DIR}"\)'
        fixed_function_match = re.search(pattern, fixed_content, re.DOTALL)
        
        if not fixed_function_match:
            print("Error: Could not find the fixed function implementation in the GitHub version")
            return False
        
        fixed_implementation = fixed_function_match.group()
        
        # Replace the old implementation with the fixed one
        # Look for the old pattern: return subprocess.call([os.path.join(LCM_BIN_DIR, name + file_extension), *args], close_fds=False)
        old_pattern = r'return subprocess\.call\(\[os\.path\.join\(LCM_BIN_DIR, name \+ file_extension\), \*args\], close_fds=False\)'
        
        if not re.search(old_pattern, original_content):
            print("Error: Could not find the old implementation pattern to replace")
            return False
        
        patched_content = re.sub(old_pattern, fixed_implementation, original_content)
        
        # Write the patched content
        with open(lcm_init_file, 'w') as f:
            f.write(patched_content)
        
        print("LCM Windows compatibility patch applied successfully!")
        return True
        
    except urllib.error.URLError as e:
        print(f"Error downloading from GitHub: {e}")
        return False
    except Exception as e:
        print(f"Error applying patch: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python apply_lcm_patch.py <path_to_lcm_init.py>")
        sys.exit(1)
    
    lcm_init_file = sys.argv[1]
    
    if not os.path.exists(lcm_init_file):
        print(f"Error: File not found: {lcm_init_file}")
        sys.exit(1)
    
    success = apply_lcm_windows_patch(lcm_init_file)
    sys.exit(0 if success else 1)
