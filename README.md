# kaggle-chemical-structure-and-logp

## download data
cd to project directory then type

```bash
mkdir -p ./data; curl -L -o ./data/archive_logp.zip https://www.kaggle.com/api/v1/datasets/download/matthewmasters/chemical-structure-and-logp; unzip ./data/archive_logp.zip -d ./data/; rm ./data/archive_logp.zip
```