# MegaSurvival World Generator

The generator of MegaSurvival's definitive 64 × 64 km world map and its tuning tool, a local web page in which each stage is tuned with sliders. The generator uses rules and noise only; nothing is painted. The spec, meaning the stages, techniques and output format, is `WORLDGEN.md` in the game repository.

- **Start:** double-click `start.cmd`. On first run it creates `.venv` and installs `requirements.txt`, then serves `http://localhost:8765`. Manual start: `.venv\Scripts\python server.py`.
- **Export:** the Export button on stage 12's card runs `worldgen/export_job.py` as its own process. It writes the full-resolution files to `../MegaSurvivalWorld/<mapId>/` and the client package to `<game repo>/Assets/StreamingAssets/World/<mapId>/`.
- **Code:** `server.py` is the HTTP server and JSON API, and `ui/` is the page. `worldgen/core/` holds the parameters, the stage base class, the pipeline and cache, and the noise and raster helpers. `worldgen/stages/` has one module per stage, registered in order in `__init__.py`.
- **Language:** everything written here is English.
