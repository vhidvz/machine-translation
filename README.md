# Quick Start

Multilingual machine translation microservice powered by the Facebook/mBART-large model, offering high-quality translations across multiple languages with contextual accuracy and fluency.

```sh
git clone git@github.com:vhidvz/machine-translation.git
cd machine-translation && docker-compose up -d
```

Endpoints are fully documented using OpenAPI Specification 3 (OAS3) at:

- ReDoc: <http://localhost:8000/redoc>
- Swagger: <http://localhost:8000/docs>

## Documentation

To generate the documentation for the python model, execute the following command:

```sh
pdoc --output-dir docs model.py
```
