import sqlite3
import time

t1 = time.time()
conn = sqlite3.connect('face.db')
c = conn.cursor()
c.execute("select * from images where created_time>='1' and created_time <= '1424246395'")
#c.execute("PRAGMA table_info(images)")
#print c.fetchall()
print time.time() - t1