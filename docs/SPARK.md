# Pandora auf der DGX Spark betreiben und über Tailscale teilen

Pandora läuft als Docker-Compose-Stack auf der Spark (`thinkstationpgx-afbe`) neben dem
vLLM-Container. Andere Personen erreichen es nur über Tailscale. An der FritzBox wird **nichts**
freigegeben.

```
Gast-Laptop ──Tailscale──▶ tailscale serve (HTTPS, setzt Tailscale-User-Login)
                              │
                              ▼  127.0.0.1:8765 (nur lokal)
                         ┌──────────────┐  internes Netz   ┌──────────────────┐
                         │   pandora    │ ───────────────▶ │     sandbox      │
                         │ (uid 10001)  │                  │ kein Netz, ro-FS │
                         └──────┬───────┘                  │ nobody, 2 GB RAM │
                                │                          └──────────────────┘
                                ▼
                     vLLM  http://192.168.2.84:8008/v1
```

- **Identität:** Tailscale Serve setzt bei jeder Anfrage die Kopfzeile `Tailscale-User-Login`.
  Das gilt auch für Personen, mit denen du die Spark geteilt hast. Kopfzeilen, die ein Client
  selbst mitschickt, entfernt Serve. Pandora lauscht nur auf `127.0.0.1`, damit niemand an Serve
  vorbei eine gefälschte Kopfzeile schicken kann.
- **Rollen:** Logins in `server.admins` haben volle Rechte. Alle anderen sind Gäste: Sie nutzen
  das Modell vom Server, können keine Prompts bearbeiten und nichts importieren, und sie sehen nur
  ihre eigenen Läufe und Berichte.
- **Python-Skripte** laufen im Container `sandbox`:
  - ohne Internet und ohne Heimnetz (internes Docker-Netz),
  - mit schreibgeschütztem Dateisystem,
  - als `nobody` ohne Capabilities,
  - mit höchstens 2 GB RAM, 2 CPUs und 128 Prozessen.
- **Warteschlange:** Läufe starten nacheinander. Gäste sehen ihre Position. Pro Person sind
  höchstens `max_runs_per_user` Läufe gleichzeitig wartend oder laufend.

## 1. Voraussetzungen auf der Spark

```bash
docker compose version          # Docker Compose v2 muss vorhanden sein
curl -s http://192.168.2.84:8008/v1/models   # vLLM antwortet, Modellname notieren
```

Tailscale installieren und anmelden, falls noch nicht geschehen:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4                 # Tailscale-IP der Spark notieren (100.x.y.z)
```

In der Tailscale-Admin-Konsole unter **DNS** müssen **MagicDNS** und **HTTPS Certificates**
aktiviert sein. Tailscale Serve braucht beides für die `https://…ts.net`-Adresse.

## 2. Pandora einrichten

```bash
git clone -b claude/focused-thompson-gp1641 https://github.com/theoalm66/agent-science
cd agent-science
cp deploy/config.spark.yaml config.yaml
nano config.yaml
```

In `config.yaml` drei Stellen anpassen (alle mit `TODO` markiert):

1. `llm.model`: der Modellname aus `curl …/v1/models`
2. `server.admins`: dein Tailscale-Login, zu finden in der Admin-Konsole unter **Users**
3. `server.allowed_origins`: die Adresse, die `tailscale serve` in Schritt 3 ausgibt

Optional legst du API-Keys in eine Datei `.env` neben `docker-compose.yml` (sie wird nicht committet):

```bash
echo 'S2_API_KEY=dein-semantic-scholar-key' > .env
chmod 600 .env
```

Starten:

```bash
docker compose up -d --build
docker compose ps                               # pandora und sandbox: running
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/api/config   # erwartet: 403
```

Die `403` ist richtig: Ohne Tailscale-Identität lässt Pandora niemanden herein. Docker startet
beide Container nach einem Neustart der Spark automatisch (`restart: unless-stopped`).

## 3. Über Tailscale freigeben

```bash
sudo tailscale serve --bg 8765
tailscale serve status          # zeigt https://thinkstationpgx-afbe.<tailnet>.ts.net
```

Die Freigabe bleibt auch nach einem Neustart bestehen. Diese Adresse trägst du in
`config.yaml` unter `server.allowed_origins` ein und lädst danach neu:
`docker compose restart pandora`. Öffne die Adresse dann auf deinem Laptop, der im selben
Tailnet ist. Oben rechts sollte das ⚙ sichtbar sein, denn du bist Admin.

## 4. Gäste einladen

1. Tailscale-Admin-Konsole → **Machines** → `thinkstationpgx-afbe` → **Share…** → E-Mail-Adresse
   eintragen.
2. Der Gast installiert Tailscale, nimmt die Einladung an und öffnet die `https://…ts.net`-Adresse.

## 5. Zugriffsregeln: Gäste dürfen nur an Pandora

Ohne Regeln könnten Gäste über Tailscale auch andere Dienste der Spark erreichen, zum Beispiel
vLLM auf Port 8008 oder SSH. Ergänze deshalb in der Admin-Konsole unter **Access controls** einen
Grant, der Gästen nur Port 443 der Spark erlaubt. Tailscale verweigert standardmäßig alles, was
nicht ausdrücklich erlaubt ist.

```jsonc
{
  "hosts": {
    "pgx-spark": "100.x.y.z"        // Tailscale-IP der Spark (tailscale ip -4)
  },
  "grants": [
    // Deine eigenen Geräte: wie bisher (passe das an deine bestehende Policy an)
    { "src": ["autogroup:member"], "dst": ["*"], "ip": ["*"] },
    // Geteilte Nutzer: nur HTTPS auf der Spark, also Tailscale Serve und damit Pandora
    { "src": ["autogroup:shared"], "dst": ["pgx-spark"], "ip": ["tcp:443"] }
  ]
}
```

Wenn deine Policy schon Regeln enthält, füge nur den zweiten Grant hinzu und prüfe, dass keine
bestehende Regel `autogroup:shared` oder `*` als Quelle für die Spark erlaubt. Der Editor in der
Admin-Konsole prüft die Syntax vor dem Speichern.

## 6. Prüfen, dass alles dicht ist

Von einem Gast-Gerät aus (oder du lädst dich mit einem zweiten Account selbst ein):

- `https://thinkstationpgx-afbe.<tailnet>.ts.net` öffnet Pandora **ohne** ⚙. Statt der
  Modellwahl steht in der Statuszeile „signed in as …“.
- `http://<tailscale-ip>:8008/v1/models` ist **nicht** erreichbar (vLLM).
- `ssh <tailscale-ip>` ist **nicht** erreichbar.
- Ein Lauf des Gastes erscheint bei dir als Admin in der Bibliothek, beim Gast nur dessen eigene
  Läufe.

## Betrieb

| Aufgabe | Befehl |
|---|---|
| Logs ansehen | `docker compose logs -f pandora` |
| Update auf neuen Stand | `git pull && docker compose up -d --build` |
| Konfiguration ändern | `config.yaml` bearbeiten, dann `docker compose restart pandora` |
| Anhalten | `docker compose down` (Daten bleiben im Volume `pandora-data`) |
| Freigabe beenden | `sudo tailscale serve --https=443 off` |

Berichte und die Versionshistorie der Prompts liegen im Docker-Volume `pandora-data` und
überstehen Updates. Prompts, die du als Admin in Pandora bearbeitest, werden in die Dateien
im Container geschrieben. Nach `docker compose up --build` gilt wieder der Stand aus dem Repo.
Deine Versionen bleiben aber in der Historie, und du kannst sie in Pandora erneut aktivieren.
Dauerhaft übernimmst du eine Änderung, indem du sie im Repo committest.
