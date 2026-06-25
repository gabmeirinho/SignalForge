import sqlite3

conn = sqlite3.connect("data/signalforge.sqlite3")
conn.row_factory = sqlite3.Row

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for table in tables:
    print(table['name'])

rows = conn.execute("""
                    SELECT f.ticker, COUNT(c.id) AS chunk_count
                    FROM filings as f
                    LEFT JOIN chunks AS c ON c.filing_id = f.id
                    GROUP BY f.ticker
                    ORDER BY chunk_count DESC
""").fetchall()

for row in rows:
    print(dict(row))

for row in conn.execute("SELECT * FROM filings WHERE ticker='INTC'"):
    print(dict(row))

# for row in conn.execute("SELECT * FROM embedding_runs"):
#     print(dict(row))



# count = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
# print(count)

# first_row = conn.execute("SELECT * FROM chunks").fetchone()
# # print(dict(first_row))

# nvidia_chunks = conn.execute("""
#                             SELECT
#                             c.id, c.section_title, c.chunk_index, c.text
#                             FROM chunks AS c
#                             JOIN filings AS f
#                                 ON c.filing_id = f.id
#                             WHERE f.ticker='NVDA'
#                             ORDER BY c.section_id, c.chunk_index
#                              """).fetchmany(2)

# # for row in nvidia_chunks:
#     # print(dict(row))

# for row in conn.execute("""
#                         SELECT section_id, section_title, COUNT(*) AS chunks
#                         FROM chunks
#                         GROUP BY section_id, section_title
#                         """):
#     print(dict(row))

# conn = sqlite3.connect("data/qdrant/collection/sec_chunks/storage.sqlite")
# conn.row_factory = sqlite3.Row

# tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
# for table in tables:
#     print(table['name'])

# cursor = conn.execute("SELECT * FROM points LIMIT 1")
# column_names = [desc[0] for desc in cursor.description]
# print(column_names)