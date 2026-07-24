# Migration guide — applying this patch to your actual repo

Your repo layout today (confirmed from this conversation):
```
~/My_projects/juice-shop/
  requirements.txt
  app/
    Dockerfile
    main.py
    embedding.py
  docker-compose.rag.yml
  config/rag.yml
  frontend/...
```

## Step 1: Back up what you have
```bash
cd ~/My_projects/juice-shop
cp -r app app.backup-$(date +%Y%m%d)
cp requirements.txt requirements.txt.backup
```

## Step 2: Unzip this patch into your repo root
Download `juice-shop-assistant-upgrade.zip` and extract it so its contents
land directly in `~/My_projects/juice-shop` (NOT into a subfolder):
```bash
cd ~/My_projects/juice-shop
unzip -o /path/to/juice-shop-assistant-upgrade.zip -d .
```
This will:
- **Replace** `app/main.py`, `app/embedding.py` → the old `embedding.py` can
  be deleted after confirming the new `app/services/embedding_service.py`
  is in place (same logic, just relocated - see README Point 13).
- **Replace** `app/Dockerfile` with the hardened version (Point 15).
- **Add** `app/config.py`, `app/logging_config.py`, `app/models.py`,
  `app/prompts.py`, `app/security.py`, `app/__init__.py`,
  `app/services/*.py`.
- **Add** `tests/` at repo root.
- **Replace** root `requirements.txt` (adds `pydantic-settings`, test deps).
- **Add** `docker-compose.additions.yml`, `.env.secrets.example`, `README.md`.

```bash
rm -f app/embedding.py   # confirm app/services/embedding_service.py exists first
```

## Step 3: Create your real secrets file
```bash
cp .env.secrets.example .env.secrets
nano .env.secrets   # fill in OPENROUTER_API_KEY (you already have this)
                     # generate INGEST_API_KEY:
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
echo ".env.secrets" >> .gitignore
```

## Step 4: Merge `docker-compose.additions.yml` into `docker-compose.rag.yml`
Open both side by side and merge the `assistant` service's `environment:`,
`env_file:`, `healthcheck:`, and `restart:` blocks from
`docker-compose.additions.yml` into your existing `assistant` service in
`docker-compose.rag.yml`. Do the same for `chroma`. Leave `juice-shop` as-is
except adding `restart: unless-stopped`.

Remove the old flat `OPENROUTER_API_KEY: "${OPENROUTER_API_KEY}"` line from
`environment:` — it's now supplied via `env_file: [.env.secrets]` instead.

## Step 5: Run the tests locally before rebuilding (fast feedback loop)
```bash
cd ~/My_projects/juice-shop
python3 -m venv .venv-test && source .venv-test/bin/activate
pip install -r requirements.txt
pytest tests/ -v
deactivate
```
All 27 should pass (verified in a clean environment while building this
patch). If any fail here, it's worth fixing before rebuilding Docker -
much faster iteration than rebuild-and-curl.

## Step 6: Re-ingest is required
The retrieval service now caches known product names for fuzzy/substring
matching, refreshed by `ingest_products_to_chroma()`. Existing Chroma data
from before this patch is still valid (same embedding approach), but the
in-memory name cache starts empty on a fresh container, so re-ingest once
after deploying:
```bash
docker compose -f docker-compose.rag.yml build assistant
docker compose -f docker-compose.rag.yml up -d --force-recreate assistant
curl -X POST http://localhost:8000/assistant/ingest \
  -H "X-API-Key: <the INGEST_API_KEY you generated>"
```
Note the new required header - the old unauthenticated
`curl -X POST http://localhost:8000/assistant/ingest` will now 401.

## Step 7: Verify readiness and health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```
`/health/ready` should show `{"ready": true, "checks": {"chroma": true, "llm_configured": true}}`.

## Step 8: Test end-to-end exactly as before
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen3-8b","messages":[{"role":"user","content":"What juice do you have?"}],"stream":true}'
```
Then test multi-turn in the same request cycle by adding a
`"conversation_id": "test-1"` field to the JSON body on two consecutive
calls - the second one referencing "it" should resolve correctly if the
first mentioned a specific product.

## Rollback
If anything goes sideways:
```bash
rm -rf app
mv app.backup-<date> app
cp requirements.txt.backup requirements.txt
docker compose -f docker-compose.rag.yml build assistant
docker compose -f docker-compose.rag.yml up -d --force-recreate assistant
```
