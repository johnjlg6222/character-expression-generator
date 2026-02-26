# Character Expression Generator

Generate consistent character expression variants from a single reference image using Google Nano Banana Pro via Replicate.

Upload a reference image, pick emotions, and get back a set of consistent variants with transparent backgrounds.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Replicate API token
```

## Run

```bash
python app.py
```

Opens at `http://localhost:7860`

## Cost

~$0.04 per image via Replicate.
