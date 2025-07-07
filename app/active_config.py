# active_config.py
from app.config import get_config, update_config

SETUP = "pronostia"
# Get the base config for the chosen setup
cfg = get_config(SETUP)

# Apply dynamic update immediately
update_config(cfg,
              bearing_used='Bearing2',
              channel='both',
              n_channels=2)
