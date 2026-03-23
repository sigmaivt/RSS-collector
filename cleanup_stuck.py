import asyncio
import sys
sys.path.insert(0, 'src')

import db

async def cleanup():
    db.set_db_path('./data/pipeline.db')
    async with db.get_conn() as conn:
        # Delete stuck outbox entries with max retries
        cursor = await conn.execute("DELETE FROM telegram_outbox WHERE sent=0 AND attempts>=5")
        deleted = cursor.rowcount
        await conn.commit()
        print(f'✅ Deleted {deleted} stuck outbox entries')
        
        if deleted > 0:
            print('These entries will be re-processed when they appear again in the pipeline')

asyncio.run(cleanup())