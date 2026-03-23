import asyncio
import sys
sys.path.insert(0, 'src')

import db

async def check():
    db.set_db_path('./data/pipeline.db')
    async with db.get_conn() as conn:
        # Get outbox entry
        async with conn.execute("SELECT * FROM telegram_outbox WHERE sent=0 LIMIT 1") as cur:
            row = await cur.fetchone()
            if row:
                entry = dict(row)
                message = entry['message_text']
                print(f'=== Message Details ===')
                print(f'Length: {len(message)} characters')
                print(f'Chat ID: {entry["chat_id"]}')
                print(f'\n=== Full Message ===')
                print(message)
                print('\n=== End of Message ===')

asyncio.run(check())