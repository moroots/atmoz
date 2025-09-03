import copy
from typing import Any, Dict, Optional

#%% 

# Dummy classes to simulate nested objects
class Tick:
    def tick_params(self, labelsize=None):
        print(f"Tick.tick_params called with labelsize={labelsize}")
        return f"tick_params({labelsize})"

class Colorbar:
    def __init__(self):
        self.ax = Tick()
    def set_label(self, label=None, size=None, weight=None):
        print(f"Colorbar.set_label called with label={label}, size={size}, weight={weight}")
        return f"set_label({label})"

class Axes:
    def set_ylabel(self, ylabel=None):
        print(f"Axes.set_ylabel called with ylabel={ylabel}")
        return f"set_ylabel({ylabel})"
    def set_xlabel(self, xlabel=None):
        print(f"Axes.set_xlabel called with xlabel={xlabel}")
        return f"set_xlabel({xlabel})"
    def grid(self, visible=None, color=None):
        print(f"Axes.grid called with visible={visible}, color={color}")
        return f"grid({visible}, {color})"

class Figure:
    def __init__(self):
        self.colorbar_obj = Colorbar()
    def colorbar(self, mappable=None, ax=None, pad=None, ticks=None):
        print(f"Figure.colorbar called with mappable={mappable}, ax={ax}, pad={pad}, ticks={ticks}")
        return self.colorbar_obj
    def savefig(self, fname=None):
        print(f"Figure.savefig called with fname={fname}")
        return f"savefig({fname})"
    

def apply_plot_params(fig, ax, params: Dict[str, Any], obj: Optional[Any] = None):
    results = {}

    for func_name, kwargs in params.items():
        # Split dotted path
        parts = func_name.split(".")
        target = obj

        # Resolve top-level object
        if target is None:
            print(f"{target=}")

            if parts[0] == "ax":
                target, parts = ax, parts[1:]
            elif parts[0] == "fig":
                target, parts = fig, parts[1:]
            else:
                raise ValueError(f"Cannot resolve target for {func_name}")

        # Traverse attributes
        for attr in parts:
            target = getattr(target, attr, None)
            if target is None:
                break

        if target is None or not callable(target):
            continue

        # Extract sub-functions
        call_kwargs = copy.deepcopy(kwargs) if isinstance(kwargs, dict) else kwargs
        sub_funcs = None
        if isinstance(call_kwargs, dict):
            sub_funcs = call_kwargs.pop("sub_functions", None)

        # Call the target function
        if isinstance(call_kwargs, dict):
            result = target(**call_kwargs)
        elif isinstance(call_kwargs, (list, tuple)):
            result = target(*call_kwargs)
        else:
            result = target(call_kwargs)

        results[func_name] = result

        # Recurse into sub-functions using the result as obj
        if sub_funcs and result is not None:
            _, _, sub_results = apply_plot_params(fig, ax, sub_funcs, obj=result)
            results.update(sub_results)

    return fig, ax, results


# ----------------------------
# Example usage
fig = Figure()
ax = Axes()

params = {
    "ax.set_ylabel": {"ylabel": "Altitude (km)"},
    "fig.colorbar": {
        "mappable": "dummy_mappable",
        "ax": ax,
        "pad": 0.01,
        "ticks": [0, 1, 2],
        "sub_functions": {
            "set_label": {"label": "Ozone", "size": 16, "weight": "bold"},
            "ax.tick_params": {"labelsize": 12}
        }
    }
}

fig, ax, _ = apply_plot_params(fig, ax, params)