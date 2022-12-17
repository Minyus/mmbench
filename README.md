# mmbench

## Set up Docker container

```bash
docker-compose up -d --build
docker exec -it mm bash
```

## CLI

### Single model config 

```bash
python tools/validate_model_configs.py configs/_base_/models/efficientnet_b0.py
```

### Multiple model configs

```bash
python tools/validate_model_configs.py configs/_base_/models/*.py
```
