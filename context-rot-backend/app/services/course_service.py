"""Authored DBMS corpus for offline integration demos; not a vector index."""
import re
from sqlalchemy import select
from app.database.database import SessionLocal
from app.database.runtime_models import CourseChunk

NOTES = [
    ("Primary and foreign keys", "A primary key uniquely identifies each row in a table. A primary key cannot contain NULL or duplicate values. A foreign key references a candidate key in another table and enforces referential integrity."),
    ("Normalization", "Normalization organizes relational tables to reduce redundancy and update anomalies. First normal form requires atomic attribute values. Second normal form removes partial dependencies on a composite candidate key. Third normal form removes transitive dependencies of non-key attributes on candidate keys."),
    ("ACID transactions", "A transaction is a logical unit of database work. ACID means Atomicity, Consistency, Isolation, and Durability. Atomicity means all operations commit or none do. Consistency preserves database constraints. Isolation controls interactions between concurrent transactions. Durability preserves committed changes after failure."),
    ("SQL joins", "An INNER JOIN returns rows that satisfy the join condition in both tables. A LEFT JOIN includes every row from the left table and matching rows from the right table, using NULL where no right-side match exists. A CROSS JOIN returns the Cartesian product."),
    ("Indexes", "A database index accelerates retrieval at the cost of storage and additional write work. A B-tree index maintains ordered keys and supports equality and range queries. An index does not always improve a query; selectivity and the query plan matter."),
    ("Concurrency and deadlocks", "Concurrency control coordinates simultaneous transactions. A serializable schedule has the effect of some serial execution. A deadlock occurs when transactions wait cyclically for resources held by each other. Detection can use a wait-for graph; a victim transaction can be rolled back."),
    ("SQL aggregation", "GROUP BY groups rows for aggregate functions such as COUNT, SUM, and AVG. WHERE filters rows before grouping. HAVING filters groups after aggregation. COUNT(*) counts rows while COUNT(column) counts non-NULL values in that column."),
    ("Relational model", "A relation is a table of tuples and attributes. A tuple is a row and an attribute is a column. A candidate key is a minimal set of attributes that uniquely identifies a tuple. A database schema describes the structure and constraints of the database."),
]
STOP = set("a an the is are of to in and or for with what how why explain please me tell about does do can you it this that from on at by as give".split())


def tokens(text: str) -> set[str]:
    return {t.rstrip("s") for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP}


def seed_course():
    with SessionLocal() as db:
        if db.scalar(select(CourseChunk.id).limit(1)) is None:
            db.add_all([CourseChunk(topic=t, text=x, page=i+1, source="Authored DBMS demo notes") for i, (t, x) in enumerate(NOTES)])
            db.commit()
        # Seed core knowledge bases if not present
        from app.database.models import KnowledgeBase, User
        if db.scalar(select(KnowledgeBase.id).limit(1)) is None:
            system_user = db.scalar(select(User).limit(1))
            if not system_user:
                system_user = User(username="system_admin", email="admin@system.local", hashed_password="dummy")
                db.add(system_user)
                db.flush()
            core_kbs = [
                KnowledgeBase(name="Biology", subject="Biology", owner_id=system_user.id),
                KnowledgeBase(name="Maths", subject="Mathematics", owner_id=system_user.id),
                KnowledgeBase(name="Physics", subject="Physics", owner_id=system_user.id),
                KnowledgeBase(name="Chemistry", subject="Chemistry", owner_id=system_user.id),
            ]
            db.add_all(core_kbs)
            db.commit()


def search(query: str, top_k=5):
    query_tokens = tokens(query)
    with SessionLocal() as db:
        ranked = []
        for row in db.scalars(select(CourseChunk)):
            overlap = query_tokens & tokens(row.topic + " " + row.text)
            if overlap:
                ranked.append(dict(text=row.text, source=row.source, page=row.page, score=len(overlap)/max(1,len(query_tokens)), topic=row.topic))
        return sorted(ranked, key=lambda x: x["score"], reverse=True)[:top_k]
