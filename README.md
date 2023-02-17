# Resolve CLI



## Install

```shell
$ python -m venv venv
# Linux
$ source venv/bin/activate
# Windows
$ venv\Scripts\Activate.ps1
$ pip install -r requirements.txt
```



## Development

```shell
pip install -e .
```



## Use

### Help

```shell
# In an activated venv
$ resolve-scoring-cli score --help
```

### Compute score

```shell
# In an activated venv
$ resolve-scoring-cli score \
    -r "legacy/tests/resources/Database_FC_proof1_ID.csv" \
    -s "legacy/tests/resources/Database_MP_proof1_ID.csv" \
    -o "C:\Temp\resolve-out.csv" \
    --biosex-context "Female bias" \
    --biosex-scenario "High" \
    -i "FC-4" -i "FC-8"
```
