"""AppContext — the config-derived state passed into every tab renderer.

Built by `ui.config.render_config()` and handed to the other renderers as an
argument. That argument dependency is what enforces "Config is read first": the
other tabs cannot render without the context, so there is no comment-guarded
ordering to get wrong.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AppContext:
    predictor: Any
    bayesian: Any
    presets: Dict[str, Any]
    costs: Dict[str, float]
    carbon_kwargs: Dict[str, Any]
    use_advanced_chemistry: bool
    exotic_strength_enabled: bool
    robust: bool
    design_age: Optional[float]
    ticket_config: Dict[str, Any]
