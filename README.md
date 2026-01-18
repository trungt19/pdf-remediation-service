# PDF Remediation Microservice

Automated PDF accessibility remediation API for the AccessibleGov Compliance Suite.

## Features

- **Metadata Fixes**: Set document title, language
- **OCR Support**: Add text layer to scanned PDFs (via ocrmypdf)
- **Accessibility Analysis**: Score PDFs for WCAG compliance

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze PDF without modifying |
| POST | `/remediate` | Fix PDF and return remediated version |
| POST | `/batch-analyze` | Analyze multiple PDFs |

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
docker build -t pdf-remediation .
docker run -p 8000:8000 pdf-remediation
```

## Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## License

Proprietary - Accessible Compliance Group
