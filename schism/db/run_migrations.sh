#!/bin/bash
# Runs migration files in lexicographic order from the migrations subdirectory.
# Sourced by docker-entrypoint-initdb.d — POSTGRES_USER and POSTGRES_DB are available as env vars.
set -e

MIGRATIONS_DIR="/docker-entrypoint-initdb.d/migrations"

for f in $(ls "$MIGRATIONS_DIR"/*.sql 2>/dev/null | sort); do
    echo "Applying migration: $f"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$f"
done
