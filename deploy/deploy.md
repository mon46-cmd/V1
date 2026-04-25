# V5 Deployment

Canonical operator doc: see `deploy/README.md`.

Shortest successful path on a real VPS:

```bash
cd ~/V1
chmod +x deploy/install.sh deploy/uninstall.sh deploy/run_nohup.sh
sudo deploy/install.sh
```

Verify:

```bash
systemctl status v5-trader.service v5-api.service --no-pager -l
curl -s http://127.0.0.1:8765/health | jq .
```

Fallback without systemd:

```bash
cd ~/V1
deploy/run_nohup.sh start
```
