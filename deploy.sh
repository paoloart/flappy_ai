#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Flappy AI - Web Client Deployment Script
# ═══════════════════════════════════════════════════════════════════════════
# Usage:
#   ./deploy.sh [DOMAIN]
#
# Default domain is vibegames.it. Run from the repo root on the server.
# Deploys the Vue.js web_client as a static site served by Caddy.
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN="${1:-vibegames.it}"
APP_DIR="$(pwd)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  🐦 Flappy AI - Web Client Deploy"
echo "═══════════════════════════════════════════════════════════"
echo "  Domain: $DOMAIN"
echo "  Dir:    $APP_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 0) Install basic dependencies
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v git >/dev/null 2>&1; then
  echo "📦 Installing git..."
  apt-get update && apt-get install -y git curl
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Install Node.js 20.x (with npm)
# ─────────────────────────────────────────────────────────────────────────────
install_node() {
  echo "📦 Installing Node.js 20.x..."
  
  # Wait for any apt locks to be released
  while fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || \
        fuser /var/lib/dpkg/lock >/dev/null 2>&1 || \
        fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "   Waiting for apt lock to be released..."
    sleep 2
  done
  
  # Remove old Node.js if present (Ubuntu's default doesn't include npm)
  apt-get remove -y nodejs npm 2>/dev/null || true
  
  # Install Node.js 20.x from NodeSource
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
  
  # Verify npm is available
  if ! command -v npm >/dev/null 2>&1; then
    echo "❌ npm not found after Node.js installation. Installing npm separately..."
    apt-get install -y npm
  fi
}

# Check if Node.js 20+ with npm is installed
if ! command -v node >/dev/null 2>&1; then
  install_node
elif ! command -v npm >/dev/null 2>&1; then
  echo "⚠️  Node.js found but npm missing. Reinstalling..."
  install_node
else
  NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
  if [ "$NODE_VERSION" -lt 18 ]; then
    echo "⚠️  Node.js version too old ($(node --version)). Upgrading..."
    install_node
  else
    echo "✓ Node.js $(node --version) with npm $(npm --version) already installed"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2) Install Docker if missing
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "📦 Installing Docker..."
  curl -fsSL https://get.docker.com | sh
else
  echo "✓ Docker $(docker --version | cut -d' ' -f3 | tr -d ',') already installed"
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "❌ Docker Compose plugin not found. Please install a recent Docker version." >&2
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3) Build the Vue.js app
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🔨 Building web_client..."
cd "$APP_DIR/web_client"

# Clean install dependencies
npm ci --prefer-offline 2>/dev/null || npm install

# Build for production
npm run build

BUILD_SIZE=$(du -sh dist 2>/dev/null | cut -f1 || echo "unknown")
echo "✓ Build complete (${BUILD_SIZE})"

cd "$APP_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Create Caddyfile (static file server with API proxy)
# ─────────────────────────────────────────────────────────────────────────────
echo "📝 Creating Caddyfile..."
cat > Caddyfile <<EOF
${DOMAIN}, www.${DOMAIN} {
    # Compression
    encode zstd gzip
    
    # Proxy API requests to the backend
    handle /api/* {
        reverse_proxy leaderboard-api:3001
    }
    
    # Serve static files from /srv
    handle {
        root * /srv
        file_server
        
        # SPA fallback: serve index.html for client-side routes
        try_files {path} /index.html
    }
    
    # Cache static assets aggressively (they have content hashes)
    @static {
        path *.js *.css *.png *.jpg *.jpeg *.gif *.ico *.svg *.woff *.woff2 *.webp *.ogg *.wav
    }
    header @static Cache-Control "public, max-age=31536000, immutable"
    
    # Don't cache HTML (for updates)
    @html {
        path *.html /
    }
    header @html Cache-Control "no-cache, no-store, must-revalidate"
}
EOF

# ─────────────────────────────────────────────────────────────────────────────
# 5) Build the leaderboard API Docker image
# ─────────────────────────────────────────────────────────────────────────────
echo "🔨 Building leaderboard API..."
cd "$APP_DIR/web_client/server"
docker build -t flappy-leaderboard-api .
cd "$APP_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 6) Create docker-compose.yml (Caddy + Leaderboard API)
# ─────────────────────────────────────────────────────────────────────────────
echo "📝 Creating docker-compose.yml..."
cat > docker-compose.yml <<EOF
# Flappy AI - Web Client + Shared Leaderboard
# AI/RL runs in browser, leaderboard is shared via API

services:
  caddy:
    image: caddy:2-alpine
    container_name: flappy-web
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - ./web_client/dist:/srv:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - leaderboard-api

  leaderboard-api:
    image: flappy-leaderboard-api
    container_name: flappy-leaderboard
    restart: unless-stopped
    environment:
      - PORT=3001
      - DATA_DIR=/app/data
    volumes:
      - leaderboard_data:/app/data

volumes:
  caddy_data:
  caddy_config:
  leaderboard_data:
EOF

# ─────────────────────────────────────────────────────────────────────────────
# 7) Backup existing leaderboard data (if any)
# ─────────────────────────────────────────────────────────────────────────────
BACKUP_DIR="$APP_DIR/backups"
mkdir -p "$BACKUP_DIR"

# Check if leaderboard container exists and has data
if docker ps -a | grep -q flappy-leaderboard; then
  echo "💾 Backing up leaderboard data..."
  BACKUP_FILE="$BACKUP_DIR/leaderboard-$(date +%Y%m%d-%H%M%S).json"
  docker cp flappy-leaderboard:/app/data/leaderboard.json "$BACKUP_FILE" 2>/dev/null && \
    echo "   ✓ Backup saved to: $BACKUP_FILE" || \
    echo "   ⚠️ No existing leaderboard data to backup"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8) Stop old FlapPyBird containers if running
# ─────────────────────────────────────────────────────────────────────────────
if [ -f "$APP_DIR/FlapPyBird/docker-compose.yml" ]; then
  echo "🛑 Stopping old FlapPyBird containers..."
  cd "$APP_DIR/FlapPyBird"
  docker compose down 2>/dev/null || true
  cd "$APP_DIR"
fi

# Stop containers but DON'T remove volumes (data persists!)
echo "🔄 Restarting containers (leaderboard data will be preserved)..."
docker compose down 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# 9) Deploy!
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🚀 Starting services..."
docker compose up -d

# Wait for containers to start
sleep 3

# Check if containers are running
WEB_RUNNING=$(docker inspect -f '{{.State.Running}}' flappy-web 2>/dev/null || echo "false")
API_RUNNING=$(docker inspect -f '{{.State.Running}}' flappy-leaderboard 2>/dev/null || echo "false")

if [ "$WEB_RUNNING" = "true" ] && [ "$API_RUNNING" = "true" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "  ✅ Deployment successful!"
  echo "═══════════════════════════════════════════════════════════"
  echo ""
  echo "  🌐 Site:       https://${DOMAIN}"
  echo "  🏆 API:        https://${DOMAIN}/api/leaderboard"
  echo ""
  echo "  📋 Logs:       docker logs -f flappy-web"
  echo "                 docker logs -f flappy-leaderboard"
  echo "  🔄 Restart:    docker compose restart"
  echo ""
  echo "  💾 Data:       Leaderboard data persists in Docker volume"
  echo "                 Backups saved to: $BACKUP_DIR/"
  echo ""
  echo "  ⚠️  To DELETE leaderboard data (manual only):"
  echo "      docker volume rm flappy_ai_leaderboard_data"
  echo ""
else
  echo ""
  echo "❌ Containers failed to start. Check logs:"
  echo "   docker logs flappy-web"
  echo "   docker logs flappy-leaderboard"
  echo ""
  echo "   Web running: $WEB_RUNNING"
  echo "   API running: $API_RUNNING"
  exit 1
fi


