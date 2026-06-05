"""
Engine3D Build System

This module provides the BuildSystem class for building Engine3D projects
into standalone executables.

Supported backends:
- pyinstaller: Easy to use, widely supported
- nuitka: Better performance, compiles to C

Usage:
    from engine3d.build import BuildSystem
    
    builder = BuildSystem(project_path=".", backend="pyinstaller")
    builder.build(onefile=True, debug=False)
"""
import os
import sys
import subprocess
import re
import glob
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


class BuildSystem:
    """Build system for Engine3D projects.
    
    This class handles building Engine3D games into standalone executables
    using either PyInstaller or Nuitka as the backend.
    
    Attributes:
        project_path: Path to the project directory
        backend: Build backend to use ("pyinstaller" or "nuitka")
        config: Build configuration loaded from pyproject.toml
    
    Example:
        >>> builder = BuildSystem(Path("."), backend="pyinstaller")
        >>> builder.build(onefile=True, debug=False)
    """
    
    def __init__(self, project_path: Path, backend: str = "pyinstaller"):
        """Initialize the build system.
        
        Args:
            project_path: Path to the project directory
            backend: Build backend to use ("pyinstaller" or "nuitka")
        """
        self.project_path = Path(project_path).resolve()
        self.backend = backend
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load build configuration from pyproject.toml.
        
        Returns:
            Dictionary containing build configuration
        """
        config_file = self.project_path / "pyproject.toml"
        if not config_file.exists():
            # Use defaults
            return {
                "entry_point": "main.py",
                "output_name": self.project_path.name,
                "icon": "",
                "include_assets": ["assets/", "scenes/", "scripts/"],
                "exclude_modules": ["pytest", "PySide6"],
            }
        
        # Parse pyproject.toml manually (no external dependency)
        content = config_file.read_text()
        
        config = {}
        # Extract tool.engine3d.build section
        match = re.search(r'\[tool\.engine3d\.build\](.*?)(?=\[|$)', content, re.DOTALL)
        if match:
            section = match.group(1)
            # Parse key = value pairs
            for line in section.strip().split('\n'):
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Parse TOML lists: ["a", "b", "c"]
                    if value.startswith('[') and value.endswith(']'):
                        inner = value[1:-1]
                        config[key] = [
                            item.strip().strip('"').strip("'")
                            for item in inner.split(',')
                            if item.strip()
                        ]
                    else:
                        config[key] = value.strip('"').strip("'")
        
        return config
    
    def build(self, onefile: bool = False, debug: bool = False) -> bool:
        """Build the project executable.
        
        Args:
            onefile: Build as single executable file
            debug: Build debug version with console window
        
        Returns:
            True if build succeeded, False otherwise
        """
        print(f"Building '{self.config.get('output_name', 'game')}' with {self.backend}...")
        
        if self.backend == "pyinstaller":
            return self._build_pyinstaller(onefile, debug)
        elif self.backend == "nuitka":
            return self._build_nuitka(onefile, debug)
        else:
            print(f"Error: Unknown backend '{self.backend}'")
            return False
    
    def _build_pyinstaller(self, onefile: bool, debug: bool) -> bool:
        """Build using PyInstaller.
        
        Args:
            onefile: Build as single executable file
            debug: Build debug version with console window
        
        Returns:
            True if build succeeded, False otherwise
        """
        try:
            import PyInstaller.__main__
        except ImportError:
            print("PyInstaller not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            import PyInstaller.__main__
        
        entry_point = self.config.get("entry_point", "main.py")
        output_name = self.config.get("output_name", "game")
        icon = self.config.get("icon", "")
        
        args = [
            entry_point,
            "--name", output_name,
            "--distpath", "dist",
            "--workpath", "build",
            "--specpath", ".",
        ]
        
        if onefile:
            args.append("--onefile")
        else:
            args.append("--onedir")
        
        if not debug:
            args.append("--windowed")
        else:
            args.append("--console")
        
        if icon:
            args.extend(["--icon", icon])
        
        # Add assets
        for asset in self.config.get("include_assets", []):
            asset_path = self.project_path / asset
            if asset_path.exists():
                args.extend(["--add-data", f"{asset}{os.pathsep}{asset}"])
        
        # Exclude modules
        for module in self.config.get("exclude_modules", []):
            args.extend(["--exclude-module", module])
        
        print(f"Running: pyinstaller {' '.join(args)}")
        
        try:
            PyInstaller.__main__.run(args)
            print(f"\n✓ Build successful!")
            print(f"  Output: {self.project_path / 'dist' / output_name}")
            return True
        except Exception as e:
            print(f"\n✗ Build failed: {e}")
            return False
    
    def _build_nuitka(self, onefile: bool, debug: bool) -> bool:
        """Build using Nuitka.
        
        Args:
            onefile: Build as single executable file
            debug: Build debug version with console window
        
        Returns:
            True if build succeeded, False otherwise
        """
        # Check if nuitka is installed
        try:
            subprocess.run([sys.executable, "-m", "nuitka", "--version"], 
                          capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Nuitka not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "nuitka"], check=True)
        
        entry_point = self.config.get("entry_point", "main.py")
        output_name = self.config.get("output_name", "game")
        
        args = [
            sys.executable, "-m", "nuitka",
            entry_point,
            "--output-dir=dist",
            "--output-filename=" + output_name,
            "--enable-plugin=pygame",
            "--include-package=engine3d",
        ]
        
        if onefile:
            args.append("--onefile")
        else:
            args.append("--standalone")
        
        if not debug:
            if os.name == 'nt':  # Windows
                args.append("--windows-disable-console")
        
        # Add assets
        for asset in self.config.get("include_assets", []):
            asset_path = self.project_path / asset
            if asset_path.exists():
                args.extend(["--include-data-dir", f"{asset}={asset}"])
        
        print(f"Running: {' '.join(args)}")
        
        try:
            subprocess.run(args, cwd=self.project_path, check=True)
            print(f"\n✓ Build successful!")
            print(f"  Output: {self.project_path / 'dist' / output_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Build failed: {e}")
            return False
    
    @staticmethod
    def clean(project_path: Path) -> None:
        """Clean build files.
        
        Removes build directories and temporary files.
        
        Args:
            project_path: Path to the project directory
        """
        print("Cleaning build files...")
        
        dirs_to_remove = ["build", "dist", "__pycache__"]
        files_to_remove = ["*.spec", "*.pyc", "*.pyo"]
        
        for dir_name in dirs_to_remove:
            path = project_path / dir_name
            if path.exists():
                shutil.rmtree(path)
                print(f"  Removed {dir_name}/")
        
        for pattern in files_to_remove:
            for file in glob.glob(str(project_path / pattern)):
                Path(file).unlink()
                print(f"  Removed {Path(file).name}")
        
        print("✓ Clean complete")


def main():
    """Command-line interface for the build system."""
    parser = argparse.ArgumentParser(
        description="Engine3D Build System"
    )
    parser.add_argument(
        "--backend",
        choices=["pyinstaller", "nuitka"],
        default="pyinstaller",
        help="Build backend to use (default: pyinstaller)"
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build as single executable file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build debug version with console"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build files"
    )
    parser.add_argument(
        "--project-path",
        default=".",
        help="Path to project directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    project_path = Path(args.project_path).resolve()
    
    if args.clean:
        BuildSystem.clean(project_path)
        return
    
    if not (project_path / "main.py").exists():
        print(f"Error: No main.py found in {project_path}")
        print("Make sure you're in a valid Engine3D project directory.")
        sys.exit(1)
    
    builder = BuildSystem(project_path, backend=args.backend)
    success = builder.build(onefile=args.onefile, debug=args.debug)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()