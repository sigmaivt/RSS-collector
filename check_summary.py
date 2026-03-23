import asyncio
import sys
sys.path.insert(0, 'src')

import db

async def check():
    db.set_db_path('./data/pipeline.db')
    item_id = '1b203d1ba1e40a48075b138a9670d60b3499e67755c91a43588c26f87430a846'
    
    async with db.get_conn() as conn:
        # Get job with summary
        async with conn.execute("SELECT * FROM channel_jobs WHERE item_id=? AND channel_id='ai_models'", (item_id,)) as cur:
            row = await cur.fetchone()
            if row:
                job = dict(row)
                print('=== Job Details ===')
                print(f'State: {job["state"]}')
                print(f'Summary length: {len(str(job["summary"]))} characters')
                print(f'\n=== Summary ===')
                print(job["summary"])
                print('\n=== End of Summary ===')
                
                # Check if summary has duplicate pattern
                summary = str(job["summary"])
                if summary[:100] in summary[100:]:
                    print('\n⚠️  WARNING: Summary appears to contain duplicated text!')
                else:
                    print('\n✓ No obvious duplication detected in first 100 chars')
            else:
                print('Job not found')

asyncio.run(check())