import asyncio
import sys
import os

sys.path.insert(0, 'src')

DB_PATH = './data/pipeline.db'

if not os.path.exists(DB_PATH):
    print(f"DB not found at {DB_PATH}")
    sys.exit(0)

import db

async def check():
    db.set_db_path(DB_PATH)
    async with db.get_conn() as conn:
        async with conn.execute('SELECT COUNT(*) as c FROM telegram_outbox WHERE sent=0') as cur:
            row = await cur.fetchone()
        print(f'Pending outbox (sent=0): {dict(row)["c"]}')

        async with conn.execute('SELECT COUNT(*) as c FROM telegram_outbox WHERE sent=1') as cur:
            row = await cur.fetchone()
        print(f'Sent outbox (sent=1): {dict(row)["c"]}')

        async with conn.execute('SELECT channel_id, chat_id, sent, attempts, last_error FROM telegram_outbox ORDER BY created_at DESC LIMIT 10') as cur:
            rows = await cur.fetchall()
        print('Latest outbox entries:')
        for r in rows:
            print(dict(r))

        async with conn.execute('SELECT state, COUNT(*) as c FROM channel_jobs GROUP BY state') as cur:
            rows = await cur.fetchall()
        print('\nJob states:')
        for r in rows:
            print(dict(r))

asyncio.run(check())
