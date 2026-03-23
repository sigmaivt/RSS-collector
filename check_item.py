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
                print('=== Outbox Entry ===')
                for k, v in entry.items():
                    if k == 'message_text':
                        print(f'{k}: {v[:200]}...' if len(v) > 200 else f'{k}: {v}')
                    else:
                        print(f'{k}: {v}')
                
                # Get item details
                async with conn.execute("SELECT * FROM items WHERE id=?", (entry['item_id'],)) as cur:
                    item = await cur.fetchone()
                    if item:
                        print('\n=== Item Details ===')
                        item_dict = dict(item)
                        for k, v in item_dict.items():
                            if k in ['title', 'content', 'description']:
                                print(f'{k}: {v[:150]}...' if len(str(v)) > 150 else f'{k}: {v}')
                            else:
                                print(f'{k}: {v}')

asyncio.run(check())