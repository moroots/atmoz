# -*- coding: utf-8 -*-
"""
Created on 2026-02-24 21:09:28

@author: Maurice Roots

Description:
     - A simple module for storing paths in research evironments
"""

#%%

from pathlib import Path
import json 
import pandas as pd

from importlib.resources import files

class PathManager:
    def __init__(self, path_file: str =  None):
        self.path_file =  files("atmoz.config").joinpath("paths.json") if not path_file else Path(path_file)

        if not self.path_file.exists():
            self._set_paths_file(self.path_file)

        self.get_paths()
        return
    
    def _set_paths_file(self, path_file: str):
        with open(path_file, "w") as f:
            json.dump({}, f)
        return
    
    def get_paths(self):
        with self.path_file.open() as f:
            paths = json.load(f)
        self.paths = {
            name: Path(path) 
                for name, path in paths.items()
            }
        return self.paths
    
    def new_paths(self, paths: dict):
        with self.path_file.open("w") as f:
            json.dump({
                name: str(path) 
                    for name, path in paths.items()
                }, 
                f
                )
        self.get_paths()
        return self.paths
    
    def new_path(self, name: str, path: str):
        paths = self.get_paths()
        paths[name] = Path(path)
        with self.path_file.open("w") as f:
            json.dump({
                name: str(path) 
                    for name, path in paths.items()
                }, 
                f
                )
        self.get_paths()
        return self.paths
    
    def get_path(self, name: str):
        paths = self.get_paths()
        return paths.get(name, None)
    
    def remove_path(self, name: str):
        paths = self.get_paths()
        if name in paths:
            del paths[name]
            with self.path_file.open("w") as f:
                json.dump({
                    name: str(path) 
                        for name, path in paths.items()
                    }, 
                    f
                    )
        self.get_paths()
        return
    
    def clear_paths(self):
        with self.path_file.open("w") as f:
            json.dump({}, f)
        self.get_paths()
        return
    