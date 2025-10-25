# master-mtdt-2526

### Setup

En etter en:

```
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

nbstripout --install

nbdime config-git --enable
```

Pass på at det er en `-e .`i requirements.txt for å installere de lokale modulene
