# 🏥 Medical Diagnosis Assistant — Qazcode 2026

AI-powered symptom → ICD-10 diagnosis system built on Kazakhstan clinical protocols.

---

## Setup
```bash
git clone https://github.com//hack-qazcode.git](https://github.com/lanNo19/hack-qazcode.git
cd hack-qazcode
```

## Build

```bash
docker build -t submission .
```

## Run

```bash
docker run -p 8080:8080 \
  -e HUB_URL=https://hub.qazcode.ai \
  -e API_KEY=<your_api_key> \
  server
```

Open **http://localhost:8080** in your browser.

---

## API

**`POST /diagnose`**

```json
// Request
{ "symptoms": "острая боль в правом подреберье, тошнота" }

// Response
{
  "diagnoses": [
    { "rank": 1, "icd10_code": "O14.1", "name": "Тяжёлая преэклампсия", "reasoning": "..." },
    { "rank": 2, "icd10_code": "K80.0", "name": "Желчнокаменная болезнь", "reasoning": "..." }
  ]
}
```

**`GET /`** — Web UI
